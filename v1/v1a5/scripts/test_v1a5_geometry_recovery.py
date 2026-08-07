from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from v1a5.geometry_recovery import (
    GeometryRecoveryConfig,
    assert_nonrecovered_geometry_unchanged,
    materialize_geometry_candidate,
)


def frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for z in range(8):
        rows.append(
            {
                "case_id": "case_x",
                "slice_idx": z,
                "image_width": 256,
                "image_height": 256,
                "left_smooth_presence": int(z >= 4),
                "right_smooth_presence": 0,
                "left_smooth_x1": 50.0 + z,
                "left_smooth_y1": 60.0,
                "left_smooth_x2": 90.0 + z,
                "left_smooth_y2": 110.0,
                "right_smooth_x1": 140.0,
                "right_smooth_y1": 60.0,
                "right_smooth_x2": 180.0,
                "right_smooth_y2": 110.0,
            }
        )
    b1 = pd.DataFrame(rows)
    r2b = b1.copy()
    r2b.loc[r2b["slice_idx"].isin([1, 2]), "left_smooth_presence"] = 1
    r2b.loc[r2b["slice_idx"].isin([1, 2]), [
        "left_smooth_x1", "left_smooth_y1", "left_smooth_x2", "left_smooth_y2"
    ]] = [5.0, 5.0, 20.0, 20.0]
    return b1, r2b


def main() -> None:
    b1, r2b = frame()

    nearest, nearest_audit = materialize_geometry_candidate(
        b1,
        r2b,
        GeometryRecoveryConfig(mode="nearest_anchor"),
    )
    assert len(nearest_audit) == 2
    assert np.allclose(
        nearest.loc[nearest["slice_idx"] == 2, [
            "left_smooth_x1", "left_smooth_y1", "left_smooth_x2", "left_smooth_y2"
        ]].iloc[0].to_numpy(dtype=float),
        b1.loc[b1["slice_idx"] == 4, [
            "left_smooth_x1", "left_smooth_y1", "left_smooth_x2", "left_smooth_y2"
        ]].iloc[0].to_numpy(dtype=float),
    )
    assert_nonrecovered_geometry_unchanged(r2b, nearest, b1)

    trend, trend_audit = materialize_geometry_candidate(
        b1,
        r2b,
        GeometryRecoveryConfig(mode="linear_trend", trend_anchor_count=4),
    )
    assert len(trend_audit) == 2
    # Anchor boxes move +1 px in x per slice, so extrapolated z=2 should recover
    # approximately the same trend rather than the deliberately corrupted native box.
    z2 = trend.loc[trend["slice_idx"] == 2].iloc[0]
    assert abs(float(z2["left_smooth_x1"]) - 52.0) < 1e-5
    assert abs(float(z2["left_smooth_x2"]) - 92.0) < 1e-5
    assert_nonrecovered_geometry_unchanged(r2b, trend, b1)

    print("V1a.5 geometry-recovery synthetic gate: PASSED")


if __name__ == "__main__":
    main()
