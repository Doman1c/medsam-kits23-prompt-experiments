from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


TARGET_NAMES = {1: "kidney", 2: "tumor", 3: "cyst"}


def window_ct(
    image: np.ndarray,
    center: float = 40.0,
    width: float = 400.0,
) -> np.ndarray:
    """Clip CT values to a window and normalize them to [0, 1]."""
    if width <= 0:
        raise ValueError("Window width must be greater than zero.")

    low = center - width / 2.0
    high = center + width / 2.0
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image.astype(np.float32)


def discover_cases(dataset_root: Path) -> list[Path]:
    """Return case folders containing both image and segmentation files."""
    return [
        case_dir
        for case_dir in sorted(dataset_root.glob("case_*"))
        if (case_dir / "imaging.nii.gz").exists()
        and (case_dir / "segmentation.nii.gz").exists()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert KiTS23 3D NIfTI volumes into windowed 2D axial "
            "NPZ slices for the MedSAM V0 experiment."
        )
    )
    parser.add_argument(
        "--kits-root",
        "--dataset-root",
        dest="dataset_root",
        type=Path,
        required=True,
        help="Directory containing KiTS23 case_xxxxx folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory containing index.csv and slices/.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Maximum valid cases to process; 0 means all local cases.",
    )
    parser.add_argument(
        "--target-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="KiTS23 labels to index. Default: 1 2 3.",
    )
    parser.add_argument(
        "--min-target-pixels",
        type=int,
        default=20,
        help="Minimum target pixels needed to index a slice-target pair.",
    )
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(
            f"KiTS23 dataset directory not found: {args.dataset_root}"
        )

    unsupported = [x for x in args.target_ids if x not in TARGET_NAMES]
    if unsupported:
        raise ValueError(
            f"Unsupported target IDs: {unsupported}; "
            f"supported IDs are {sorted(TARGET_NAMES)}."
        )

    case_dirs = discover_cases(args.dataset_root)
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]
    if not case_dirs:
        raise RuntimeError(
            "No complete cases found. Each case needs imaging.nii.gz "
            "and segmentation.nii.gz."
        )

    slices_dir = args.output_root / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    processed_cases = 0
    saved_slices = 0
    label_inventory: dict[str, list[int]] = {}

    for case_dir in tqdm(case_dirs, desc="Preparing KiTS23 2D data"):
        case_id = case_dir.name
        try:
            image_nii = nib.load(str(case_dir / "imaging.nii.gz"))
            label_nii = nib.load(str(case_dir / "segmentation.nii.gz"))

            image_volume = image_nii.get_fdata(dtype=np.float32)
            label_volume = np.rint(
                label_nii.get_fdata(dtype=np.float32)
            ).astype(np.uint8)

            if image_volume.shape != label_volume.shape:
                raise ValueError(
                    f"Shape mismatch: image={image_volume.shape}, "
                    f"label={label_volume.shape}"
                )
            if image_volume.ndim != 3:
                raise ValueError(f"Expected 3D volume, got {image_volume.shape}")

            label_inventory[case_id] = [
                int(value) for value in np.unique(label_volume)
            ]

            for slice_index in range(image_volume.shape[2]):
                label_slice = label_volume[:, :, slice_index]
                target_records: list[tuple[int, int]] = []

                for target_id in args.target_ids:
                    pixels = int(np.count_nonzero(label_slice == target_id))
                    if pixels >= args.min_target_pixels:
                        target_records.append((target_id, pixels))

                if not target_records:
                    continue

                image_slice = window_ct(
                    image_volume[:, :, slice_index],
                    center=args.window_center,
                    width=args.window_width,
                )

                relative_path = (
                    Path("slices") / f"{case_id}_z{slice_index:04d}.npz"
                )
                output_path = args.output_root / relative_path

                if args.overwrite or not output_path.exists():
                    np.savez_compressed(
                        output_path,
                        image=image_slice.astype(np.float32),
                        label=label_slice.astype(np.uint8),
                    )

                saved_slices += 1
                for target_id, pixels in target_records:
                    rows.append(
                        {
                            "case_id": case_id,
                            "slice_index": slice_index,
                            "npz_path": relative_path.as_posix(),
                            "target_id": target_id,
                            "target_name": TARGET_NAMES[target_id],
                            "target_pixels": pixels,
                            "image_height": int(image_slice.shape[0]),
                            "image_width": int(image_slice.shape[1]),
                        }
                    )

            processed_cases += 1

        except Exception as error:
            print(
                f"\n[skip] {case_id}: "
                f"{type(error).__name__}: {error}"
            )

    if not rows:
        raise RuntimeError(
            "No slice-target records were produced. Check labels and "
            "--min-target-pixels."
        )

    index = pd.DataFrame(rows).sort_values(
        ["case_id", "slice_index", "target_id"]
    )
    index.to_csv(args.output_root / "index.csv", index=False)

    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "processed_cases": processed_cases,
        "saved_positive_slices": saved_slices,
        "indexed_slice_target_records": len(index),
        "target_ids": args.target_ids,
        "target_names": {str(k): v for k, v in TARGET_NAMES.items()},
        "min_target_pixels": args.min_target_pixels,
        "window_center": args.window_center,
        "window_width": args.window_width,
        "label_inventory": label_inventory,
    }
    with (args.output_root / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nPreparation complete.")
    print("Processed cases:", processed_cases)
    print("Saved positive slices:", saved_slices)
    print("Indexed slice-target records:", len(index))
    print("\nRecords by target:")
    print(index.groupby("target_name").size().sort_values(ascending=False))
    print("\nOutput:", args.output_root)


if __name__ == "__main__":
    main()
