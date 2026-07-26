from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

FORMAT_VERSION = "v1a1_dual_yx_v2"


def migrate_case(case_dir: Path) -> tuple[int, bool]:
    metadata_path = case_dir / "case_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("processed_format_version") == FORMAT_VERSION:
        return 0, False

    files = sorted(case_dir.glob("slice_*.npz"))
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            image = np.asarray(data["image"], dtype=np.float32)
            label = np.asarray(data["label"], dtype=np.uint8)
        if image.ndim != 2 or label.shape != image.shape:
            raise ValueError(f"Invalid NPZ shapes in {path}: {image.shape}, {label.shape}")
        image = np.ascontiguousarray(image.T)
        label = np.ascontiguousarray(label.T)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, image=image, label=label)
            handle.flush()
        temporary.replace(path)

    old_shape = metadata.get("slice_shape_yx")
    if isinstance(old_shape, list) and len(old_shape) == 2:
        metadata["slice_shape_yx"] = [int(old_shape[1]), int(old_shape[0])]
    metadata["processed_format_version"] = FORMAT_VERSION
    metadata["volume_array_layout"] = "ZYX"
    metadata["slice_array_layout"] = "YX"
    metadata["anatomical_lr_axis_in_slice"] = "columns"
    metadata["anatomical_side_rule"] = (
        "low column = left kidney; high column = right kidney"
    )
    metadata["migration_note"] = "Transposed legacy axial slices from (X,Y) to (Y,X)."
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return len(files), True


def create_cache(processed_root: Path, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, "w") as archive:
        archive.add(processed_root, arcname="processed")
    temporary.replace(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy v1a.1 processed NPZ slices from (X,Y) to (Y,X)"
    )
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--cache-output", default=None)
    args = parser.parse_args()

    root = Path(args.processed_root)
    case_dirs = sorted(p for p in root.glob("case_*" ) if p.is_dir())
    if not case_dirs:
        raise RuntimeError(f"No case directories under {root}")

    migrated_cases = 0
    migrated_files = 0
    for case_dir in tqdm(case_dirs, desc="Migrating axial layout XY -> YX"):
        count, changed = migrate_case(case_dir)
        migrated_files += count
        migrated_cases += int(changed)

    print(f"Migrated cases: {migrated_cases}/{len(case_dirs)}")
    print(f"Transposed NPZ files: {migrated_files}")
    print(f"Processed format: {FORMAT_VERSION}")

    if args.cache_output:
        cache_path = Path(args.cache_output)
        print(f"Writing corrected cache: {cache_path}")
        create_cache(root, cache_path)


if __name__ == "__main__":
    main()
