from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


CASE_RE = re.compile(r"case_\d+")


def window_ct(image: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    image = np.clip(image, low, high)
    return ((image - low) / max(high - low, 1e-8)).astype(np.float32)


def find_cases(root: Path) -> list[tuple[str, Path, Path]]:
    cases = []
    for imaging in root.rglob("imaging.nii.gz"):
        segmentation = imaging.parent / "segmentation.nii.gz"
        if not segmentation.exists():
            continue
        match = CASE_RE.search(str(imaging.parent))
        case_id = match.group(0) if match else imaging.parent.name
        cases.append((case_id, imaging, segmentation))
    return sorted(cases, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize KiTS23 and save axial 2D NPZ slices")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = find_cases(Path(args.raw_root))
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError(f"No KiTS23 cases found under {args.raw_root}")

    for case_id, image_path, label_path in tqdm(cases, desc="Preparing canonical cases"):
        case_output = output_root / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        image_original = nib.load(str(image_path))
        label_original = nib.load(str(label_path))
        image_nii = nib.as_closest_canonical(image_original)
        label_nii = nib.as_closest_canonical(label_original)
        if tuple(nib.aff2axcodes(image_nii.affine)) != ("R", "A", "S"):
            raise RuntimeError(f"Canonical orientation is not RAS for {case_id}")
        image = np.asarray(image_nii.dataobj, dtype=np.float32)
        label = np.asarray(label_nii.dataobj, dtype=np.int16)
        if image.shape != label.shape or image.ndim != 3:
            raise ValueError(f"Shape mismatch in {case_id}: {image.shape} vs {label.shape}")
        # Canonical NIfTI is stored as (X, Y, Z). For a conventional axial
        # NumPy image, use rows=Y and columns=X, therefore reorder to (Z, Y, X).
        # This is important because anatomical left/right is then represented by
        # the 2D column coordinate. In RAS, increasing X points right, so low
        # column indices correspond to the anatomical left kidney.
        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))
        image = window_ct(image, args.window_center, args.window_width)

        metadata = {
            "case_id": case_id,
            "original_image_axcodes": "".join(nib.aff2axcodes(image_original.affine)),
            "original_label_axcodes": "".join(nib.aff2axcodes(label_original.affine)),
            "canonical_axcodes": "RAS",
            "canonical_shape_xyz": list(image_nii.shape),
            "slice_shape_yx": [int(image.shape[1]), int(image.shape[2])],
            "num_slices": int(image.shape[0]),
            "processed_format_version": "v1a1_dual_yx_v2",
            "volume_array_layout": "ZYX",
            "slice_array_layout": "YX",
            "anatomical_lr_axis_in_slice": "columns",
            "anatomical_side_rule": "low column = left kidney; high column = right kidney",
            "window_center": args.window_center,
            "window_width": args.window_width,
        }
        (case_output / "case_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        for z in range(image.shape[0]):
            output_path = case_output / f"slice_{z:04d}.npz"
            if output_path.exists() and not args.overwrite:
                try:
                    if output_path.stat().st_size <= 0:
                        raise EOFError("empty NPZ")
                    with np.load(output_path, allow_pickle=False) as existing:
                        if "image" not in existing or "label" not in existing:
                            raise KeyError("missing image/label")
                    continue
                except Exception:
                    output_path.unlink(missing_ok=True)
            temporary_path = output_path.with_name(output_path.name + ".tmp")
            with temporary_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    image=image[z].astype(np.float32),
                    label=label[z].astype(np.uint8),
                )
                handle.flush()
            temporary_path.replace(output_path)

    print(f"Prepared {len(cases)} canonical RAS cases at {output_root}")
    print(f"Total NPZ slices: {sum(1 for _ in output_root.rglob('*.npz'))}")


if __name__ == "__main__":
    main()
