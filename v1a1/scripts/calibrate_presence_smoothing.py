from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.calibration import (
    AREA_BINS,
    SIDES,
    CalibrationConfig,
    add_threshold_independent_smoothing,
    apply_calibration,
    parse_float_grid,
    parse_int_grid,
    save_config,
)
from v1a1.metrics import evaluate_prediction_frame
from v1a1.utils import parse_int_list


def geometry_table(frame: pd.DataFrame, variant: str, expansion_scale: float, target_labels: tuple[int, ...]) -> pd.DataFrame:
    geometry_frame = frame.copy()
    for side in SIDES:
        geometry_frame[f"{side}_{variant}_presence"] = 1
    _, details = evaluate_prediction_frame(
        geometry_frame,
        variant=variant,
        expansion_scale=expansion_scale,
        target_labels=target_labels,
    )
    return details


def presence_vector(frame: pd.DataFrame, config: CalibrationConfig) -> tuple[np.ndarray, float, int]:
    calibrated = apply_calibration(frame, config)
    values = []
    for side in SIDES:
        side_rows = calibrated[["case_id", "slice_idx", f"{side}_{config.variant}_presence"]].copy()
        side_rows["side"] = side
        side_rows.rename(columns={f"{side}_{config.variant}_presence": "pred_present"}, inplace=True)
        values.append(side_rows)
    long = pd.concat(values, ignore_index=True)
    long["side_order"] = long["side"].map({"left": 0, "right": 1})
    long.sort_values(["case_id", "slice_idx", "side_order"], inplace=True)
    pred_vector = long["pred_present"].to_numpy(dtype=np.uint8)
    left_present = calibrated[f"left_{config.variant}_presence"].to_numpy(dtype=np.uint8)
    right_present = calibrated[f"right_{config.variant}_presence"].to_numpy(dtype=np.uint8)
    both = (left_present == 1) & (right_present == 1)
    left_x = calibrated[f"left_{config.variant}_center_x_norm"].to_numpy(dtype=float)
    right_x = calibrated[f"right_{config.variant}_center_x_norm"].to_numpy(dtype=float)
    violations = both & (left_x >= right_x)
    violation_rate = float(violations.sum() / max(int(both.sum()), 1))
    return pred_vector, violation_rate, int(both.sum())


def summarize(
    base_long: pd.DataFrame,
    geometry: pd.DataFrame,
    pred_present: np.ndarray,
    config: CalibrationConfig,
    expansion_scale: float,
    order_violation_rate: float,
    both_predicted_slices: int,
) -> dict:
    details = base_long.copy()
    details["pred_present"] = pred_present
    positive = details["gt_present"] == 1
    negative = ~positive
    tp = int((positive & (details.pred_present == 1)).sum())
    fn = int((positive & (details.pred_present == 0)).sum())
    fp = int((negative & (details.pred_present == 1)).sum())
    tn = int((negative & (details.pred_present == 0)).sum())

    geo = geometry[[
        "case_id", "slice_idx", "side", "center_error_pixels", "center_error_fraction",
        "tight_box_iou", "expanded_coverage", "box_purity",
    ]].copy()
    details = details.merge(geo, on=["case_id", "slice_idx", "side"], how="left", validate="one_to_one")
    present_positive = positive & (details.pred_present == 1)
    absent_positive = positive & (details.pred_present == 0)
    for column in ("tight_box_iou", "expanded_coverage", "box_purity"):
        details.loc[absent_positive, column] = 0.0
    details.loc[absent_positive, "center_error_pixels"] = np.inf
    details.loc[absent_positive, "center_error_fraction"] = np.inf
    details["prompt_success"] = (
        positive
        & (details.pred_present == 1)
        & (details.center_error_fraction <= 0.10)
        & (details.expanded_coverage >= 0.95)
    ).astype(np.uint8)

    positive_details = details[positive].copy()
    valid_center = positive_details.center_error_fraction.replace([np.inf, -np.inf], np.nan).dropna()

    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)
    summary = {
        **config.to_dict(),
        "expansion_scale": float(expansion_scale),
        "presence_accuracy": float(accuracy),
        "positive_recall": float(recall),
        "false_positive_rate": float(fpr),
        "center_error_pixels_median": float(positive_details.center_error_pixels.replace(np.inf, np.nan).median()),
        "center_error_fraction_mean": float(valid_center.mean()) if len(valid_center) else float("nan"),
        "center_error_fraction_median": float(valid_center.median()) if len(valid_center) else float("nan"),
        "center_within_05": float((positive_details.center_error_fraction <= 0.05).mean()),
        "center_within_10": float((positive_details.center_error_fraction <= 0.10).mean()),
        "center_over_20": float((positive_details.center_error_fraction > 0.20).mean()),
        "tight_box_iou": float(positive_details.tight_box_iou.mean()),
        "expanded_coverage_mean": float(positive_details.expanded_coverage.mean()),
        "box_purity_mean": float(positive_details.box_purity.mean()),
        "prompt_success_rate": float(positive_details.prompt_success.mean()),
        "side_order_violation_rate": float(order_violation_rate),
        "positive_side_samples": int(len(positive_details)),
        "both_predicted_slices": int(both_predicted_slices),
    }
    for side in SIDES:
        subset = positive_details[positive_details.side == side]
        summary[f"{side}_positive_recall"] = float((subset.pred_present == 1).mean()) if len(subset) else float("nan")
        summary[f"{side}_prompt_success"] = float(subset.prompt_success.mean()) if len(subset) else float("nan")
    for area_bin in AREA_BINS:
        subset = positive_details[positive_details.area_bin == area_bin]
        summary[f"prompt_success_{area_bin}"] = float(subset.prompt_success.mean()) if len(subset) else float("nan")
        summary[f"coverage_{area_bin}"] = float(subset.expanded_coverage.mean()) if len(subset) else float("nan")

    center_quality = 1.0 - float(
        np.clip(positive_details.center_error_fraction.replace(np.inf, 1.0), 0.0, 1.0).mean()
    )
    summary["center_quality"] = center_quality
    summary["continuous_score"] = float(
        0.35 * summary["expanded_coverage_mean"]
        + 0.25 * center_quality
        + 0.20 * summary["positive_recall"]
        + 0.10 * summary["tight_box_iou"]
        - 0.10 * summary["false_positive_rate"]
        - 0.10 * summary["side_order_violation_rate"]
    )
    summary["recall_aware_score"] = float(
        0.40 * summary["prompt_success_rate"]
        + 0.25 * summary["continuous_score"]
        + 0.25 * summary["positive_recall"]
        - 0.10 * summary["false_positive_rate"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only sweep of independent left/right thresholds and z-axis smoothing"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--left-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--right-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--max-gaps", default="0,1,2")
    parser.add_argument("--include-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expansion-scale", type=float, default=1.15)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--min-overall-recall", type=float, default=0.95)
    parser.add_argument("--min-side-recall", type=float, default=0.90)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_threshold_independent_smoothing(frame)

    base_parts = []
    for side in SIDES:
        part = frame[["case_id", "slice_idx", f"{side}_gt_presence", f"{side}_area_bin"]].copy()
        part["side"] = side
        part.rename(
            columns={f"{side}_gt_presence": "gt_present", f"{side}_area_bin": "area_bin"},
            inplace=True,
        )
        base_parts.append(part)
    base_long = pd.concat(base_parts, ignore_index=True)
    base_long["side_order"] = base_long.side.map({"left": 0, "right": 1})
    base_long.sort_values(["case_id", "slice_idx", "side_order"], inplace=True)
    base_long.drop(columns="side_order", inplace=True)
    base_long.reset_index(drop=True, inplace=True)

    target_labels = parse_int_list(args.target_labels)
    print("Precomputing raw geometry (one pass over validation NPZ files)...")
    raw_geometry = geometry_table(frame, "raw", args.expansion_scale, target_labels)
    print("Precomputing smoothed geometry (one pass over validation NPZ files)...")
    smooth_geometry = geometry_table(frame, "smooth", args.expansion_scale, target_labels)

    left_thresholds = parse_float_grid(args.left_thresholds)
    right_thresholds = parse_float_grid(args.right_thresholds)
    max_gaps = parse_int_grid(args.max_gaps)
    configurations: list[CalibrationConfig] = []
    if args.include_raw:
        configurations.extend(
            CalibrationConfig("raw", -1, left, right)
            for left, right in product(left_thresholds, right_thresholds)
        )
    configurations.extend(
        CalibrationConfig("smooth", gap, left, right)
        for gap, left, right in product(max_gaps, left_thresholds, right_thresholds)
    )

    rows = []
    for index, config in enumerate(configurations, start=1):
        if index == 1 or index % 25 == 0 or index == len(configurations):
            print(f"Calibration {index}/{len(configurations)}: {config}")
        pred, order_violation_rate, both_predicted_slices = presence_vector(frame, config)
        geometry = raw_geometry if config.variant == "raw" else smooth_geometry
        rows.append(summarize(
            base_long, geometry, pred, config, args.expansion_scale,
            order_violation_rate, both_predicted_slices,
        ))

    result = pd.DataFrame(rows)
    result["meets_recall_constraints"] = (
        (result.positive_recall >= args.min_overall_recall)
        & (result.left_positive_recall >= args.min_side_recall)
        & (result.right_positive_recall >= args.min_side_recall)
    )
    feasible = result[result.meets_recall_constraints].copy()
    if len(feasible):
        ranked = feasible.sort_values(
            ["prompt_success_rate", "continuous_score", "false_positive_rate", "positive_recall"],
            ascending=[False, False, True, False],
        )
        constraints_met = True
    else:
        ranked = result.sort_values(
            ["recall_aware_score", "positive_recall", "prompt_success_rate", "false_positive_rate"],
            ascending=[False, False, False, True],
        )
        constraints_met = False

    result.sort_values(
        ["meets_recall_constraints", "prompt_success_rate", "continuous_score", "positive_recall"],
        ascending=[False, False, False, False],
    ).to_csv(output_dir / "presence_smoothing_sweep.csv", index=False)
    ranked.head(20).to_csv(output_dir / "top20_presence_smoothing.csv", index=False)

    best = ranked.iloc[0].to_dict()
    config = CalibrationConfig(
        variant=str(best["variant"]),
        max_gap=int(best["max_gap"]),
        left_threshold=float(best["left_threshold"]),
        right_threshold=float(best["right_threshold"]),
    )
    constraints = {
        "min_overall_recall": args.min_overall_recall,
        "min_side_recall": args.min_side_recall,
    }
    clean_metrics = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in best.items()
        if key not in {"variant", "max_gap", "left_threshold", "right_threshold"}
    }
    save_config(
        output_dir / "recommended_calibration.json",
        config,
        clean_metrics,
        constraints,
        constraints_met,
    )
    calibrated = apply_calibration(frame, config)
    calibrated.to_csv(output_dir / "recommended_predictions.csv", index=False)

    print("\nRecommended validation calibration:")
    print(json.dumps({"configuration": config.to_dict(), "metrics": clean_metrics}, indent=2, ensure_ascii=False))
    if not constraints_met:
        print("WARNING: No configuration met all recall constraints; fallback ranking was used.")


if __name__ == "__main__":
    main()
