from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a6.geometry_consensus import ConsensusConfig, apply_consensus, assert_invariants


def validate_validation_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    frame_keys = set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))
    val_keys = set(zip(val["case_id"].astype(str), val["slice_idx"].astype(int)))
    if frame_keys != val_keys:
        raise RuntimeError("Prediction rows do not exactly match validation split")
    if int(val["case_id"].nunique()) != 20 or len(val) != 3644:
        raise RuntimeError(
            f"Unexpected validation split: cases={val['case_id'].nunique()}, rows={len(val)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize frozen V1a.6 Stage-1 candidates")
    parser.add_argument("--b1-predictions", required=True)
    parser.add_argument("--r2b-predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    b1 = pd.read_csv(args.b1_predictions, dtype={"case_id": str})
    r2b = pd.read_csv(args.r2b_predictions, dtype={"case_id": str})
    validate_validation_rows(b1, Path(args.manifest))
    validate_validation_rows(r2b, Path(args.manifest))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "g1_r2b_native": ConsensusConfig(mode="native"),
        "g4_satellite_median": ConsensusConfig(mode="satellite_median"),
        "g5_rolling_median": ConsensusConfig(mode="rolling_median", rolling_window=5),
        "g6_robust_linear": ConsensusConfig(
            mode="robust_linear", mad_threshold=3.5, minimum_inliers=5
        ),
    }

    candidate_specs = []
    recovered_reference = None
    for name, config in configs.items():
        frame, recovered = apply_consensus(r2b, b1, config)
        invariant = assert_invariants(frame, r2b, b1)
        candidate_dir = output_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = candidate_dir / "recommended_predictions.csv"
        frame.to_csv(prediction_path, index=False)

        if recovered_reference is None:
            recovered_reference = recovered.copy()
            recovered_reference.to_csv(output_dir / "recovered_side_keys.csv", index=False)
        else:
            key_columns = ["case_id", "slice_idx", "side", "segment_number"]
            if not recovered_reference[key_columns].equals(recovered[key_columns]):
                raise RuntimeError(f"Recovered key set changed for {name}")

        config_payload = {
            "name": name,
            "status": "validation_only_v1a6_stage1_candidate",
            "test_set_used": False,
            "decoder": {
                "mode": config.mode,
                "rolling_window": config.rolling_window,
                "mad_threshold": config.mad_threshold,
                "minimum_inliers": config.minimum_inliers,
            },
            "invariants": invariant,
        }
        (candidate_dir / "decoder_config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        calibration = {
            "configuration": {
                "variant": "smooth",
                "left_threshold": 0.0035,
                "right_threshold": 0.04,
                "max_gap": 0,
            },
            "status": "v1a6_materialized_r2b_presence_with_local_geometry",
            "test_set_used": False,
        }
        calibration_path = candidate_dir / "recommended_calibration.json"
        calibration_path.write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        candidate_specs.append(
            {
                "name": name,
                "directory": str(candidate_dir),
                "predictions": str(prediction_path),
                "calibration_json": str(calibration_path),
                "decoder": config_payload["decoder"],
                "invariants": invariant,
                "expansion_scale": 1.20,
            }
        )

    recovered_reference = recovered_reference if recovered_reference is not None else pd.DataFrame()
    summary = {
        "status": "validation_only_v1a6_stage1_candidates",
        "test_set_used": False,
        "validation_cases": 20,
        "validation_rows": 3644,
        "recovered_side_slices": int(len(recovered_reference)),
        "recovered_gt_positive_side_slices": int(
            recovered_reference.get("gt_present", pd.Series(dtype=int)).astype(int).sum()
        ),
        "candidates": candidate_specs,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
