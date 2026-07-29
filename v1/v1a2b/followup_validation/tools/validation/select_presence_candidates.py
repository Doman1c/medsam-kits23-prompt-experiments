from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CONFIG_KEYS = ("variant", "max_gap", "left_threshold", "right_threshold")


def config_tuple(row: pd.Series) -> tuple:
    return tuple(row[key] for key in CONFIG_KEYS)


def choose_unique(selected: list[tuple[str, pd.Series]], label: str, row: pd.Series) -> None:
    existing = {config_tuple(item) for _, item in selected}
    if config_tuple(row) not in existing:
        selected.append((label, row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select diverse feasible presence-calibration candidates for direct MedSAM comparison.")
    parser.add_argument("--v1a1-root", required=True)
    parser.add_argument("--sweep-csv", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-overall-recall", type=float, default=0.95)
    parser.add_argument("--min-side-recall", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    sys.path.insert(0, args.v1a1_root)
    from v1a1.calibration import CalibrationConfig, add_threshold_independent_smoothing, apply_calibration, save_config

    sweep = pd.read_csv(args.sweep_csv)
    base = pd.read_csv(args.base_predictions)
    feasible = sweep[
        (sweep["positive_recall"] >= args.min_overall_recall)
        & (sweep["left_positive_recall"] >= args.min_side_recall)
        & (sweep["right_positive_recall"] >= args.min_side_recall)
    ].copy()
    if feasible.empty:
        raise RuntimeError("No calibration configuration meets all recall constraints.")

    feasible["recall_distance"] = (feasible["positive_recall"] - args.target_recall).abs()
    selected: list[tuple[str, pd.Series]] = []

    choose_unique(selected, "prompt_best", feasible.sort_values(
        ["prompt_success_rate", "continuous_score", "false_positive_rate", "positive_recall"],
        ascending=[False, False, True, False],
    ).iloc[0])
    choose_unique(selected, "fpr_best", feasible.sort_values(
        ["false_positive_rate", "prompt_success_rate", "continuous_score"],
        ascending=[True, False, False],
    ).iloc[0])
    choose_unique(selected, "target_recall", feasible.sort_values(
        ["recall_distance", "false_positive_rate", "prompt_success_rate", "continuous_score"],
        ascending=[True, True, False, False],
    ).iloc[0])
    choose_unique(selected, "continuous_best", feasible.sort_values(
        ["continuous_score", "prompt_success_rate", "false_positive_rate"],
        ascending=[False, False, True],
    ).iloc[0])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoothed = add_threshold_independent_smoothing(base)
    manifest = []

    for label, row in selected:
        candidate_dir = output_dir / label
        candidate_dir.mkdir(parents=True, exist_ok=True)
        config = CalibrationConfig(
            variant=str(row["variant"]),
            max_gap=int(row["max_gap"]),
            left_threshold=float(row["left_threshold"]),
            right_threshold=float(row["right_threshold"]),
        )
        metrics = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in row.to_dict().items()
            if key not in CONFIG_KEYS
        }
        constraints = {
            "min_overall_recall": args.min_overall_recall,
            "min_side_recall": args.min_side_recall,
        }
        config_path = candidate_dir / "recommended_calibration.json"
        prediction_path = candidate_dir / "recommended_predictions.csv"
        save_config(config_path, config, metrics, constraints, True)
        apply_calibration(smoothed, config).to_csv(prediction_path, index=False)
        manifest.append({
            "name": label,
            "directory": str(candidate_dir),
            "calibration_json": str(config_path),
            "predictions_csv": str(prediction_path),
            **config.to_dict(),
            "positive_recall": float(row["positive_recall"]),
            "left_positive_recall": float(row["left_positive_recall"]),
            "right_positive_recall": float(row["right_positive_recall"]),
            "false_positive_rate": float(row["false_positive_rate"]),
            "prompt_success_rate": float(row["prompt_success_rate"]),
            "continuous_score": float(row["continuous_score"]),
        })

    pd.DataFrame(manifest).to_csv(output_dir / "candidate_manifest.csv", index=False)
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps({"candidates": manifest}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
