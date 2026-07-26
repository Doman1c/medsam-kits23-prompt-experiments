from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


CASE_RE = re.compile(r"case_\d+")


def infer_case_id(path: Path) -> str:
    match = CASE_RE.search(str(path))
    return match.group(0) if match else path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KiTS23 NIfTI orientation before canonicalization")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imaging_paths = sorted(raw_root.rglob("imaging.nii.gz"))
    if not imaging_paths:
        raise RuntimeError(f"No imaging.nii.gz files under {raw_root}")

    rows = []
    for imaging_path in tqdm(imaging_paths, desc="Auditing orientation"):
        segmentation_path = imaging_path.parent / "segmentation.nii.gz"
        if not segmentation_path.exists():
            continue
        image = nib.load(str(imaging_path))
        label = nib.load(str(segmentation_path))
        image_axcodes = tuple(str(x) for x in nib.aff2axcodes(image.affine))
        label_axcodes = tuple(str(x) for x in nib.aff2axcodes(label.affine))
        canonical = nib.as_closest_canonical(image)
        rows.append(
            {
                "case_id": infer_case_id(imaging_path),
                "image_path": str(imaging_path.resolve()),
                "segmentation_path": str(segmentation_path.resolve()),
                "image_shape": "x".join(map(str, image.shape)),
                "label_shape": "x".join(map(str, label.shape)),
                "image_axcodes": "".join(image_axcodes),
                "label_axcodes": "".join(label_axcodes),
                "canonical_axcodes": "".join(map(str, nib.aff2axcodes(canonical.affine))),
                "affine_match": bool(np.allclose(image.affine, label.affine, atol=1e-4)),
                "shape_match": bool(image.shape == label.shape),
                "voxel_sizes": ",".join(f"{x:.6g}" for x in image.header.get_zooms()[:3]),
                "affine_determinant": float(np.linalg.det(image.affine[:3, :3])),
            }
        )

    frame = pd.DataFrame(rows).sort_values("case_id")
    frame.to_csv(output_dir / "orientation_audit.csv", index=False)
    summary = {
        "cases": int(len(frame)),
        "original_orientation_counts": frame["image_axcodes"].value_counts().to_dict(),
        "label_orientation_counts": frame["label_axcodes"].value_counts().to_dict(),
        "canonical_orientation_counts": frame["canonical_axcodes"].value_counts().to_dict(),
        "shape_mismatch_cases": frame.loc[~frame["shape_match"], "case_id"].tolist(),
        "affine_mismatch_cases": frame.loc[~frame["affine_match"], "case_id"].tolist(),
        "all_canonical_ras": bool((frame["canonical_axcodes"] == "RAS").all()),
    }
    (output_dir / "orientation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["shape_mismatch_cases"]:
        raise RuntimeError("Image/segmentation shape mismatch detected")


if __name__ == "__main__":
    main()
