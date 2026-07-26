from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.calibration import add_threshold_independent_smoothing, apply_calibration, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen validation calibration to val/test base predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    frame = add_threshold_independent_smoothing(frame)
    config = load_config(args.calibration_json)
    calibrated = apply_calibration(frame, config)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output, index=False)
    print(f"Applied {config.to_dict()} to {len(calibrated)} rows")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
