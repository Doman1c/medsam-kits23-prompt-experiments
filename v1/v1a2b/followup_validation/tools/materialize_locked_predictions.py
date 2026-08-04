from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a locked validation calibration configuration to a raw dual-kidney prediction CSV.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root / "v1" / "v1a1"))
    from v1a1.calibration import CalibrationConfig, add_threshold_independent_smoothing, apply_calibration, save_config

    raw_path = Path(args.raw_predictions)
    locked_path = Path(args.locked_config)
    output_dir = Path(args.output_dir)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not locked_path.is_file():
        raise FileNotFoundError(locked_path)

    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    config = CalibrationConfig(
        variant=str(locked["variant"]),
        max_gap=int(locked.get("max_gap", 0)),
        left_threshold=float(locked["left_threshold"]),
        right_threshold=float(locked["right_threshold"]),
    )
    raw = pd.read_csv(raw_path)
    calibrated = apply_calibration(add_threshold_independent_smoothing(raw), config)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_path = output_dir / "recommended_predictions.csv"
    config_path = output_dir / "recommended_calibration.json"
    calibrated.to_csv(calibrated_path, index=False)
    save_config(
        config_path,
        config,
        metrics=locked.get("presence_metrics", {}),
        constraints={"source": "locked_validation_configuration"},
        feasible=True,
    )
    print(json.dumps(config.to_dict(), indent=2))
    print("Rows:", len(calibrated))
    print("Predictions:", calibrated_path)
    print("Calibration:", config_path)


if __name__ == "__main__":
    main()
