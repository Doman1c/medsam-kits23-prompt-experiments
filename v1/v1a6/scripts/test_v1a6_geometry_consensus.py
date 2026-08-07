from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a6.geometry_consensus import ConsensusConfig, apply_consensus, assert_invariants


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for z in range(8):
        row = {
            "case_id": "case_demo",
            "slice_idx": z,
            "image_width": 512,
            "image_height": 512,
            "left_gt_presence": int(1 <= z <= 5),
            "right_gt_presence": 0,
            "left_smooth_presence_probability": 0.02 if 1 <= z <= 5 else 0.0,
            "right_smooth_presence_probability": 0.0,
            "left_smooth_presence": int(1 <= z <= 5),
            "right_smooth_presence": 0,
        }
        for side in ("left", "right"):
            if side == "left":
                cx = 120.0 + z
                cy = 200.0
                width = 60.0
                height = 80.0
                if z == 3:
                    cx = 400.0
                    cy = 400.0
            else:
                cx, cy, width, height = 360.0, 200.0, 60.0, 80.0
            row[f"{side}_smooth_x1"] = cx - width / 2
            row[f"{side}_smooth_y1"] = cy - height / 2
            row[f"{side}_smooth_x2"] = cx + width / 2
            row[f"{side}_smooth_y2"] = cy + height / 2
            row[f"{side}_gt_x1"] = row[f"{side}_smooth_x1"]
            row[f"{side}_gt_y1"] = row[f"{side}_smooth_y1"]
            row[f"{side}_gt_x2"] = row[f"{side}_smooth_x2"]
            row[f"{side}_gt_y2"] = row[f"{side}_smooth_y2"]
        rows.append(row)

    r2b = pd.DataFrame(rows)
    b1 = r2b.copy()
    b1["left_smooth_presence"] = 0
    return r2b, b1


def main() -> None:
    r2b, b1 = make_frames()
    for config in (
        ConsensusConfig(mode="native"),
        ConsensusConfig(mode="satellite_median"),
        ConsensusConfig(mode="rolling_median", rolling_window=5),
        ConsensusConfig(mode="robust_linear", mad_threshold=3.5, minimum_inliers=5),
    ):
        candidate, recovered = apply_consensus(r2b, b1, config)
        invariant = assert_invariants(candidate, r2b, b1)
        assert invariant["presence_identical_to_r2b"]
        assert invariant["changed_non_recovered_geometry_values"] == 0
        assert len(recovered) == 5

    median_candidate, _ = apply_consensus(
        r2b, b1, ConsensusConfig(mode="satellite_median")
    )
    native_outlier_cx = 0.5 * (
        r2b.loc[3, "left_smooth_x1"] + r2b.loc[3, "left_smooth_x2"]
    )
    median_outlier_cx = 0.5 * (
        median_candidate.loc[3, "left_smooth_x1"]
        + median_candidate.loc[3, "left_smooth_x2"]
    )
    assert abs(median_outlier_cx - 123.0) < abs(native_outlier_cx - 123.0)

    print("V1a.6 satellite-local geometry synthetic gate: PASSED")


if __name__ == "__main__":
    main()
