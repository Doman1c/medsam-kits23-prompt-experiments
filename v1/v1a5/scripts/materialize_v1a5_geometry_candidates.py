from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a5.geometry_recovery import (
    GeometryRecoveryConfig,
    SIDES,
    assert_nonrecovered_geometry_unchanged,
    materialize_geometry_candidate,
    recovered_side_mask,
)


def validate_validation_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    keys = set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))
    val_keys = set(zip(val["case_id"].astype(str), val["slice_idx"].astype(int)))
    if keys != val_keys:
        raise RuntimeError("Prediction rows do not exactly match validation split")
    if int(val["case_id"].nunique()) != 20 or len(val) != 3644:
        raise RuntimeError(
            f"Unexpected validation split: cases={val['case_id'].nunique()}, rows={len(val)}"
        )


def write_candidate(
    root: Path,
    name: str,
    frame: pd.DataFrame,
    payload: dict,
) -> dict:
    directory = root / "candidates" / name
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / "recommended_predictions.csv"
    frame.to_csv(prediction_path, index=False)
    calibration = {
        "configuration": {
            "variant": "smooth",
            "max_gap": 0,
            "left_threshold": 0.0035,
            "right_threshold": 0.04,
        },
        "geometry_recovery": payload,
        "status": "v1a5_materialized_geometry_candidate",
        "test_set_used": False,
    }
    calibration_path = directory / "recommended_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "geometry_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "name": name,
        "directory": str(directory),
        "predictions": str(prediction_path),
        "calibration_json": str(calibration_path),
        "geometry": payload,
        "expansion_scale": 1.20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize V1a.5 validation geometry candidates")
    parser.add_argument("--b1-predictions", required=True)
    parser.add_argument("--r2b-predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trend-anchor-count", type=int, default=5)
    args = parser.parse_args()

    b1 = pd.read_csv(args.b1_predictions, dtype={"case_id": str})
    r2b = pd.read_csv(args.r2b_predictions, dtype={"case_id": str})
    validate_validation_rows(b1, Path(args.manifest))
    validate_validation_rows(r2b, Path(args.manifest))

    b1 = b1.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
    r2b = r2b.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)

    recovered_rows = []
    for side in SIDES:
        mask = recovered_side_mask(b1, r2b, side=side)
        for index in np.flatnonzero(mask):
            recovered_rows.append(
                {
                    "case_id": str(r2b.loc[index, "case_id"]),
                    "slice_idx": int(r2b.loc[index, "slice_idx"]),
                    "side": side,
                    "gt_present": int(r2b.loc[index, f"{side}_gt_presence"]),
                    "presence_probability": float(r2b.loc[index, f"{side}_smooth_presence_probability"]),
                }
            )
    recovered = pd.DataFrame(recovered_rows).sort_values(
        ["case_id", "slice_idx", "side"]
    ).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(output_dir / "recovered_side_keys.csv", index=False)

    candidate_specs = []
    candidate_specs.append(
        write_candidate(
            output_dir,
            "g0_frozen_b1",
            b1,
            {
                "mode": "frozen_b1_reference",
                "presence_source": "frozen_b1",
                "recovered_geometry_changes": 0,
            },
        )
    )
    candidate_specs.append(
        write_candidate(
            output_dir,
            "g1_r2b_native",
            r2b,
            {
                "mode": "native",
                "presence_source": "r2b_remote_satellite",
                "recovered_geometry_changes": 0,
            },
        )
    )

    for name, config in (
        ("g2_nearest_anchor", GeometryRecoveryConfig(mode="nearest_anchor")),
        (
            "g3_linear_trend",
            GeometryRecoveryConfig(
                mode="linear_trend",
                trend_anchor_count=int(args.trend_anchor_count),
            ),
        ),
    ):
        candidate, audit = materialize_geometry_candidate(b1, r2b, config)
        assert_nonrecovered_geometry_unchanged(r2b, candidate, b1)
        directory = output_dir / "candidates" / name
        directory.mkdir(parents=True, exist_ok=True)
        audit.to_csv(directory / "geometry_recovery_audit.csv", index=False)
        candidate_specs.append(
            write_candidate(
                output_dir,
                name,
                candidate,
                {
                    "mode": config.mode,
                    "presence_source": "r2b_remote_satellite",
                    "trend_anchor_count": int(config.trend_anchor_count),
                    "recovered_side_samples": int(len(audit)),
                    "non_recovered_geometry_unchanged": True,
                    "presence_unchanged_from_r2b": True,
                },
            )
        )

    manifest = {
        "experiment": "V1a.5",
        "stage": "stage1_validation_geometry_recovery",
        "status": "materialized_before_prompt_evaluation",
        "test_set_used": False,
        "source_b1_predictions": str(Path(args.b1_predictions)),
        "source_r2b_predictions": str(Path(args.r2b_predictions)),
        "recovered_side_samples": int(len(recovered)),
        "recovered_gt_positive": int(recovered["gt_present"].sum()) if len(recovered) else 0,
        "recovered_gt_negative": int((recovered["gt_present"] == 0).sum()) if len(recovered) else 0,
        "candidates": candidate_specs,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
