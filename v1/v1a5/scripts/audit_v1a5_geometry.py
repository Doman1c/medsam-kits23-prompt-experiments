from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def geometry_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pcx, pcy = 0.5 * (pred[0] + pred[2]), 0.5 * (pred[1] + pred[3])
    gcx, gcy = 0.5 * (gt[0] + gt[2]), 0.5 * (gt[1] + gt[3])
    gw = max(float(gt[2] - gt[0]), 1.0)
    gh = max(float(gt[3] - gt[1]), 1.0)
    center_pixels = float(np.hypot(pcx - gcx, pcy - gcy))
    center_fraction = float(max(abs(pcx - gcx) / gw, abs(pcy - gcy) / gh))
    return {
        "center_error_pixels": center_pixels,
        "center_error_fraction": center_fraction,
        "tight_box_iou": box_iou(pred, gt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V1a.5 recovered-slice geometry")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--recovered-keys", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.candidate_manifest).read_text(encoding="utf-8"))
    recovered = pd.read_csv(args.recovered_keys, dtype={"case_id": str})
    recovered = recovered[recovered["gt_present"].astype(int) == 1].copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    for candidate in manifest["candidates"]:
        frame = pd.read_csv(candidate["predictions"], dtype={"case_id": str})
        frame = frame.set_index(["case_id", "slice_idx"], drop=False)
        for _, key in recovered.iterrows():
            case_id = str(key["case_id"])
            slice_idx = int(key["slice_idx"])
            side = str(key["side"])
            row = frame.loc[(case_id, slice_idx)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pred = np.asarray([float(row[f"{side}_smooth_{k}"]) for k in ("x1", "y1", "x2", "y2")])
            gt = np.asarray([float(row[f"{side}_gt_{k}"]) for k in ("x1", "y1", "x2", "y2")])
            values = geometry_metrics(pred, gt)
            detail_rows.append(
                {
                    "candidate": candidate["name"],
                    "case_id": case_id,
                    "slice_idx": slice_idx,
                    "side": side,
                    **values,
                }
            )

    details = pd.DataFrame(detail_rows)
    details.to_csv(output_dir / "recovered_geometry_details.csv", index=False)
    summary = (
        details.groupby("candidate", as_index=False)
        .agg(
            recovered_side_samples=("tight_box_iou", "size"),
            center_error_pixels_mean=("center_error_pixels", "mean"),
            center_error_pixels_median=("center_error_pixels", "median"),
            center_error_fraction_mean=("center_error_fraction", "mean"),
            center_error_fraction_median=("center_error_fraction", "median"),
            tight_box_iou_mean=("tight_box_iou", "mean"),
            tight_box_iou_median=("tight_box_iou", "median"),
        )
        .sort_values("tight_box_iou_mean", ascending=False)
    )
    summary.to_csv(output_dir / "recovered_geometry_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
