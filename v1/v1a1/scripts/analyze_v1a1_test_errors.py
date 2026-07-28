from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "v1a1").exists():
    sys.path.insert(0, str(ROOT))
elif (Path.cwd() / "v1a1").exists():
    sys.path.insert(0, str(Path.cwd()))

from v1a1.data import load_npz  # noqa: E402
from v1a1.inference import expand_box  # noqa: E402


AREA_ORDER = ["00_10", "10_50", "50_90", "90_100"]
SIDES = ("left", "right")
EXPANSION_SCALE = 1.20

# These thresholds only label descriptive error-analysis groups.
# They are NOT model thresholds and must not be used to retune the frozen test result.
LOW_DICE = 0.50
VERY_LOW_DICE = 0.20
ACCEPTABLE_DICE = 0.80
GOOD_ORACLE = 0.85
ORACLE_LIMITED = 0.75
LARGE_GAP = 0.30
CENTER_BAD = 0.20
COVERAGE_BAD = 0.80
COVERAGE_GOOD = 0.95
IOU_BAD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descriptive error analysis for the frozen V1a.1 test evaluation. "
            "This script performs no inference and no parameter tuning."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prompt-details", required=True)
    parser.add_argument("--medsam-per-slice", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-count", type=int, default=4)
    return parser.parse_args()


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else float("nan")


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.median()) if values.notna().any() else float("nan")


def validate_inputs(
    predictions: pd.DataFrame,
    prompt_details: pd.DataFrame,
    medsam: pd.DataFrame,
) -> None:
    prediction_required = {
        "case_id",
        "slice_idx",
        "npz_path",
        "image_height",
        "image_width",
        "combined_area_bin",
        "combined_relative_area",
        "left_gt_presence",
        "right_gt_presence",
        "left_smooth_presence",
        "right_smooth_presence",
    }
    prompt_required = {
        "case_id",
        "slice_idx",
        "side",
        "gt_present",
        "pred_present",
        "presence_probability",
        "center_error_fraction",
        "tight_box_iou",
        "expanded_coverage",
        "box_purity",
        "prompt_success",
    }
    medsam_required = {
        "case_id",
        "slice_idx",
        "automatic_dice",
        "automatic_iou",
        "oracle_scaled_dice",
        "automatic_to_oracle_gap",
        "combined_area_bin",
        "combined_relative_area",
    }

    for name, frame, required in (
        ("predictions", predictions, prediction_required),
        ("prompt details", prompt_details, prompt_required),
        ("MedSAM per-slice", medsam, medsam_required),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    if predictions.duplicated(["case_id", "slice_idx"]).any():
        raise ValueError("Duplicate prediction rows")
    if medsam.duplicated(["case_id", "slice_idx"]).any():
        raise ValueError("Duplicate MedSAM per-slice rows")
    if prompt_details.duplicated(["case_id", "slice_idx", "side"]).any():
        raise ValueError("Duplicate prompt-detail side rows")


def side_status(row: pd.Series) -> str:
    fp = int(row["false_positive_sides"])
    fn = int(row["false_negative_sides"])
    if fp == 0 and fn == 0:
        return "exact_sides"
    if fp > 0 and fn == 0:
        return "extra_side"
    if fp == 0 and fn > 0:
        return "missing_side"
    return "mixed_side_error"


def primary_failure(row: pd.Series) -> str:
    auto = float(row["automatic_dice"])
    oracle = float(row["oracle_scaled_dice"])
    gap = float(row["automatic_to_oracle_gap"])

    if auto >= ACCEPTABLE_DICE:
        return "acceptable"
    if auto >= LOW_DICE:
        return "moderate"

    # Low-Dice slices: assign one primary descriptive failure mechanism.
    if int(row["false_negative_sides"]) > 0:
        return "missed_side"
    if int(row["false_positive_sides"]) > 0:
        return "absent_side_false_positive"

    max_center = finite_or_nan(row["max_center_error_fraction"])
    min_coverage = finite_or_nan(row["min_expanded_coverage"])
    mean_iou = finite_or_nan(row["mean_tight_box_iou"])

    if (
        (np.isfinite(max_center) and max_center > CENTER_BAD)
        or (np.isfinite(min_coverage) and min_coverage < COVERAGE_BAD)
    ):
        return "localization_or_coverage_failure"

    if (
        (np.isfinite(mean_iou) and mean_iou < IOU_BAD)
        or (np.isfinite(min_coverage) and min_coverage < COVERAGE_GOOD)
    ):
        return "box_geometry_failure"

    if oracle < ORACLE_LIMITED and gap < 0.15:
        return "oracle_limited"

    if oracle >= GOOD_ORACLE and gap >= LARGE_GAP:
        return "high_gap_despite_acceptable_prompt_metrics"

    if oracle < ORACLE_LIMITED:
        return "oracle_limited"

    return "other_low_dice"


def build_side_aggregate(prompt_details: pd.DataFrame) -> pd.DataFrame:
    prompt = prompt_details.copy()
    prompt["case_id"] = prompt["case_id"].astype(str)
    prompt["slice_idx"] = prompt["slice_idx"].astype(int)

    positive = prompt[prompt["gt_present"].astype(int) == 1].copy()
    positive["center_error_fraction_clean"] = pd.to_numeric(
        positive["center_error_fraction"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    positive_agg = (
        positive.groupby(["case_id", "slice_idx"], as_index=False)
        .agg(
            gt_positive_sides_from_details=("side", "size"),
            mean_center_error_fraction=("center_error_fraction_clean", "mean"),
            max_center_error_fraction=("center_error_fraction_clean", "max"),
            mean_tight_box_iou=("tight_box_iou", "mean"),
            min_tight_box_iou=("tight_box_iou", "min"),
            mean_expanded_coverage=("expanded_coverage", "mean"),
            min_expanded_coverage=("expanded_coverage", "min"),
            mean_box_purity=("box_purity", "mean"),
            all_positive_prompts_success=("prompt_success", "min"),
            mean_positive_presence_probability=("presence_probability", "mean"),
        )
    )

    side_wide = prompt.pivot(
        index=["case_id", "slice_idx"],
        columns="side",
        values=["gt_present", "pred_present", "presence_probability"],
    )
    side_wide.columns = [f"{side}_{metric}" for metric, side in side_wide.columns]
    side_wide = side_wide.reset_index()

    return side_wide.merge(
        positive_agg,
        on=["case_id", "slice_idx"],
        how="left",
        validate="one_to_one",
    )


def build_slice_table(
    predictions: pd.DataFrame,
    prompt_details: pd.DataFrame,
    medsam: pd.DataFrame,
) -> pd.DataFrame:
    pred = predictions.copy()
    pred["case_id"] = pred["case_id"].astype(str)
    pred["slice_idx"] = pred["slice_idx"].astype(int)

    scores = medsam.copy()
    scores["case_id"] = scores["case_id"].astype(str)
    scores["slice_idx"] = scores["slice_idx"].astype(int)

    positive_mask = (
        pred["left_gt_presence"].astype(int).eq(1)
        | pred["right_gt_presence"].astype(int).eq(1)
    )
    pred = pred[positive_mask].copy()

    side_agg = build_side_aggregate(prompt_details)

    table = scores.merge(
        pred,
        on=["case_id", "slice_idx"],
        how="inner",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    table = table.merge(
        side_agg,
        on=["case_id", "slice_idx"],
        how="left",
        validate="one_to_one",
    )

    if len(table) != len(scores):
        raise ValueError(
            f"Merged slice table has {len(table)} rows but MedSAM table has {len(scores)}"
        )

    table["gt_prompt_count"] = (
        table["left_gt_presence"].astype(int)
        + table["right_gt_presence"].astype(int)
    )
    table["pred_prompt_count"] = (
        table["left_smooth_presence"].astype(int)
        + table["right_smooth_presence"].astype(int)
    )
    table["false_positive_sides"] = (
        (
            table["left_gt_presence"].astype(int).eq(0)
            & table["left_smooth_presence"].astype(int).eq(1)
        ).astype(int)
        + (
            table["right_gt_presence"].astype(int).eq(0)
            & table["right_smooth_presence"].astype(int).eq(1)
        ).astype(int)
    )
    table["false_negative_sides"] = (
        (
            table["left_gt_presence"].astype(int).eq(1)
            & table["left_smooth_presence"].astype(int).eq(0)
        ).astype(int)
        + (
            table["right_gt_presence"].astype(int).eq(1)
            & table["right_smooth_presence"].astype(int).eq(0)
        ).astype(int)
    )
    table["side_status"] = table.apply(side_status, axis=1)

    table["positive_oracle_gap"] = table["automatic_to_oracle_gap"].clip(lower=0.0)
    table["low_dice"] = table["automatic_dice"] < LOW_DICE
    table["very_low_dice"] = table["automatic_dice"] < VERY_LOW_DICE
    table["acceptable_dice"] = table["automatic_dice"] >= ACCEPTABLE_DICE
    table["oracle_limited_flag"] = table["oracle_scaled_dice"] < ORACLE_LIMITED
    table["large_gap_flag"] = table["automatic_to_oracle_gap"] >= LARGE_GAP
    table["localization_bad_flag"] = (
        (table["max_center_error_fraction"] > CENTER_BAD)
        | (table["min_expanded_coverage"] < COVERAGE_BAD)
    )
    table["box_geometry_bad_flag"] = (
        (table["mean_tight_box_iou"] < IOU_BAD)
        | (table["min_expanded_coverage"] < COVERAGE_GOOD)
    )
    table["primary_category"] = table.apply(primary_failure, axis=1)

    order_map = {name: index for index, name in enumerate(AREA_ORDER)}
    table["_area_order"] = table["combined_area_bin"].map(order_map)
    table = table.sort_values(
        ["_area_order", "case_id", "slice_idx"]
    ).drop(columns="_area_order").reset_index(drop=True)
    return table


def grouped_summary(
    frame: pd.DataFrame,
    group_column: str,
    low_only_denominator: bool = False,
) -> pd.DataFrame:
    records = []
    denominator = int(frame["low_dice"].sum()) if low_only_denominator else len(frame)

    for group, subset in frame.groupby(group_column, dropna=False):
        n = len(subset)
        low_n = int(subset["low_dice"].sum())
        records.append(
            {
                group_column: group,
                "n_slices": n,
                "percent_of_all_slices": n / max(len(frame), 1),
                "percent_of_low_dice_slices": (
                    low_n / max(int(frame["low_dice"].sum()), 1)
                ),
                "automatic_dice_mean": float(subset["automatic_dice"].mean()),
                "automatic_dice_median": float(subset["automatic_dice"].median()),
                "oracle_dice_mean": float(subset["oracle_scaled_dice"].mean()),
                "oracle_gap_mean": float(subset["automatic_to_oracle_gap"].mean()),
                "low_dice_rate": low_n / max(n, 1),
                "very_low_dice_rate": float(subset["very_low_dice"].mean()),
                "extra_side_rate": float((subset["false_positive_sides"] > 0).mean()),
                "missing_side_rate": float((subset["false_negative_sides"] > 0).mean()),
            }
        )
    return pd.DataFrame(records)


def build_case_summary(frame: pd.DataFrame) -> pd.DataFrame:
    case = (
        frame.groupby("case_id", as_index=False)
        .agg(
            positive_slices=("slice_idx", "size"),
            automatic_dice_mean=("automatic_dice", "mean"),
            automatic_dice_median=("automatic_dice", "median"),
            oracle_dice_mean=("oracle_scaled_dice", "mean"),
            oracle_gap_mean=("automatic_to_oracle_gap", "mean"),
            positive_gap_sum=("positive_oracle_gap", "sum"),
            low_dice_rate=("low_dice", "mean"),
            very_low_dice_rate=("very_low_dice", "mean"),
            acceptable_dice_rate=("acceptable_dice", "mean"),
            extra_side_rate=("false_positive_sides", lambda s: float((s > 0).mean())),
            missing_side_rate=("false_negative_sides", lambda s: float((s > 0).mean())),
            oracle_limited_rate=("oracle_limited_flag", "mean"),
            localization_bad_rate=("localization_bad_flag", "mean"),
            box_geometry_bad_rate=("box_geometry_bad_flag", "mean"),
            mean_relative_area=("combined_relative_area", "mean"),
        )
    )

    category_counts = pd.crosstab(frame["case_id"], frame["primary_category"])
    category_counts.columns = [f"category_count__{column}" for column in category_counts.columns]
    category_counts = category_counts.reset_index()

    case = case.merge(category_counts, on="case_id", how="left")
    case = case.sort_values(
        ["automatic_dice_mean", "oracle_gap_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return case


def build_area_summary(frame: pd.DataFrame) -> pd.DataFrame:
    area = grouped_summary(frame, "combined_area_bin")
    order_map = {name: index for index, name in enumerate(AREA_ORDER)}
    area["_order"] = area["combined_area_bin"].map(order_map)
    return area.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def single_gt_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    single = frame[frame["gt_prompt_count"] == 1].copy()
    subset = single[single["side_status"].isin(["exact_sides", "extra_side"])].copy()
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for area in AREA_ORDER:
        area_frame = subset[subset["combined_area_bin"] == area]
        for status in ("exact_sides", "extra_side"):
            group = area_frame[area_frame["side_status"] == status]
            if group.empty:
                continue
            rows.append(
                {
                    "combined_area_bin": area,
                    "side_status": status,
                    "n_slices": len(group),
                    "automatic_dice_mean": float(group["automatic_dice"].mean()),
                    "automatic_dice_median": float(group["automatic_dice"].median()),
                    "oracle_dice_mean": float(group["oracle_scaled_dice"].mean()),
                    "oracle_gap_mean": float(group["automatic_to_oracle_gap"].mean()),
                    "low_dice_rate": float(group["low_dice"].mean()),
                }
            )
    return pd.DataFrame(rows)


def correlations(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    candidates = {
        "combined_relative_area": "combined_relative_area",
        "oracle_scaled_dice": "oracle_scaled_dice",
        "oracle_gap": "automatic_to_oracle_gap",
        "max_center_error_fraction": "max_center_error_fraction",
        "mean_tight_box_iou": "mean_tight_box_iou",
        "min_expanded_coverage": "min_expanded_coverage",
        "mean_box_purity": "mean_box_purity",
        "false_positive_sides": "false_positive_sides",
        "false_negative_sides": "false_negative_sides",
    }
    result: dict[str, dict[str, float | int]] = {}

    for label, column in candidates.items():
        pair = frame[["automatic_dice", column]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(pair) < 3 or pair[column].nunique() < 2:
            result[label] = {"n": int(len(pair)), "spearman_r": float("nan"), "p_value": float("nan")}
            continue
        rho, p_value = spearmanr(pair["automatic_dice"], pair[column])
        result[label] = {
            "n": int(len(pair)),
            "spearman_r": float(rho),
            "p_value": float(p_value),
        }
    return result


def build_summary(
    frame: pd.DataFrame,
    case_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    status_summary: pd.DataFrame,
    single_gt: pd.DataFrame,
) -> dict[str, Any]:
    total_gap = float(frame["positive_oracle_gap"].sum())
    top_cases = case_summary.nlargest(5, "positive_gap_sum")
    top5_gap_fraction = (
        float(top_cases["positive_gap_sum"].sum() / total_gap)
        if total_gap > 0
        else float("nan")
    )

    single_exact = frame[
        (frame["gt_prompt_count"] == 1) & (frame["side_status"] == "exact_sides")
    ]
    single_extra = frame[
        (frame["gt_prompt_count"] == 1) & (frame["side_status"] == "extra_side")
    ]

    return {
        "status": "descriptive_test_error_analysis",
        "analysis_only_not_for_test_tuning": True,
        "definitions": {
            "low_dice": LOW_DICE,
            "very_low_dice": VERY_LOW_DICE,
            "acceptable_dice": ACCEPTABLE_DICE,
            "good_oracle": GOOD_ORACLE,
            "oracle_limited": ORACLE_LIMITED,
            "large_oracle_gap": LARGE_GAP,
            "bad_center_error_fraction": CENTER_BAD,
            "bad_coverage": COVERAGE_BAD,
            "good_coverage": COVERAGE_GOOD,
            "bad_tight_box_iou": IOU_BAD,
        },
        "overall": {
            "positive_slices": int(len(frame)),
            "low_dice_slices": int(frame["low_dice"].sum()),
            "low_dice_rate": float(frame["low_dice"].mean()),
            "very_low_dice_slices": int(frame["very_low_dice"].sum()),
            "very_low_dice_rate": float(frame["very_low_dice"].mean()),
            "acceptable_dice_rate": float(frame["acceptable_dice"].mean()),
            "extra_side_slices": int((frame["false_positive_sides"] > 0).sum()),
            "extra_side_rate": float((frame["false_positive_sides"] > 0).mean()),
            "missing_side_slices": int((frame["false_negative_sides"] > 0).sum()),
            "missing_side_rate": float((frame["false_negative_sides"] > 0).mean()),
            "large_gap_slices": int(frame["large_gap_flag"].sum()),
            "large_gap_rate": float(frame["large_gap_flag"].mean()),
            "oracle_limited_slices": int(frame["oracle_limited_flag"].sum()),
            "oracle_limited_rate": float(frame["oracle_limited_flag"].mean()),
        },
        "failure_concentration": {
            "total_positive_oracle_gap_sum": total_gap,
            "top_5_cases_gap_fraction": top5_gap_fraction,
            "top_5_cases": top_cases[
                [
                    "case_id",
                    "automatic_dice_mean",
                    "oracle_dice_mean",
                    "oracle_gap_mean",
                    "positive_gap_sum",
                ]
            ].to_dict(orient="records"),
        },
        "single_gt_extra_side_comparison": {
            "exact_side_n": int(len(single_exact)),
            "exact_side_dice_mean": (
                float(single_exact["automatic_dice"].mean()) if len(single_exact) else float("nan")
            ),
            "extra_side_n": int(len(single_extra)),
            "extra_side_dice_mean": (
                float(single_extra["automatic_dice"].mean()) if len(single_extra) else float("nan")
            ),
            "unadjusted_mean_difference_extra_minus_exact": (
                float(single_extra["automatic_dice"].mean() - single_exact["automatic_dice"].mean())
                if len(single_extra) and len(single_exact)
                else float("nan")
            ),
            "interpretation_warning": (
                "This is descriptive, not causal; case mix and relative-area distribution may differ."
            ),
        },
        "correlations_with_automatic_dice": correlations(frame),
        "primary_category_summary": category_summary.to_dict(orient="records"),
        "side_status_summary": status_summary.to_dict(orient="records"),
        "single_gt_area_stratified": single_gt.to_dict(orient="records"),
        "research_guardrail": (
            "Use these findings to formulate V1a.2 hypotheses only. "
            "Do not select thresholds, loss weights, expansion, checkpoints, or architecture variants on this test set."
        ),
    }


def plot_histogram(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame["automatic_dice"], bins=30)
    ax.axvline(frame["automatic_dice"].mean(), linestyle="--", label="mean")
    ax.axvline(frame["automatic_dice"].median(), linestyle=":", label="median")
    ax.set_xlabel("Automatic MedSAM Dice")
    ax.set_ylabel("Number of slices")
    ax.set_title("Frozen test Dice distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_auto_vs_oracle(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["oracle_scaled_dice"], frame["automatic_dice"], alpha=0.45, s=14)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Oracle-scaled MedSAM Dice")
    ax.set_ylabel("Automatic MedSAM Dice")
    ax.set_title("Automatic versus oracle Dice")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_by_area(frame: pd.DataFrame, output: Path) -> None:
    groups = [
        frame.loc[frame["combined_area_bin"] == area, "automatic_dice"].to_numpy()
        for area in AREA_ORDER
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, labels=AREA_ORDER, showfliers=False)
    ax.set_xlabel("Combined relative-area bin")
    ax.set_ylabel("Automatic MedSAM Dice")
    ax.set_title("Dice distribution by relative area")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_by_side_status(frame: pd.DataFrame, output: Path) -> None:
    order = ["exact_sides", "extra_side", "missing_side", "mixed_side_error"]
    available = [status for status in order if (frame["side_status"] == status).any()]
    groups = [
        frame.loc[frame["side_status"] == status, "automatic_dice"].to_numpy()
        for status in available
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, labels=available, showfliers=False)
    ax.set_xlabel("Predicted side status")
    ax.set_ylabel("Automatic MedSAM Dice")
    ax.set_title("Dice distribution by side-presence error")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_case_dice(case_summary: pd.DataFrame, output: Path) -> None:
    ordered = case_summary.sort_values("automatic_dice_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered["automatic_dice_mean"], label="automatic")
    ax.scatter(ordered["oracle_dice_mean"], positions, marker="x", label="oracle")
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["case_id"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Case mean Dice")
    ax.set_ylabel("Case")
    ax.set_title("Case-level automatic and oracle Dice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_prompt_overlay(
    prediction_row: pd.Series,
    analysis_row: pd.Series,
    output: Path,
) -> None:
    image, label = load_npz(prediction_row["npz_path"])
    gt = label == 1

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray")
    if gt.any():
        ax.contour(gt.astype(float), levels=[0.5], linewidths=1.0)

    for side in SIDES:
        if int(prediction_row[f"{side}_smooth_presence"]) != 1:
            continue
        box = tuple(
            float(prediction_row[f"{side}_smooth_{key}"])
            for key in ("x1", "y1", "x2", "y2")
        )
        box = expand_box(
            box,
            EXPANSION_SCALE,
            int(prediction_row["image_width"]),
            int(prediction_row["image_height"]),
        )
        x1, y1, x2, y2 = box
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.5,
            )
        )
        ax.text(x1, max(0, y1 - 3), side, fontsize=8)

    ax.set_title(
        f"{analysis_row['primary_category']} | {analysis_row['case_id']} "
        f"z={int(analysis_row['slice_idx'])}\n"
        f"auto={analysis_row['automatic_dice']:.3f}, "
        f"oracle={analysis_row['oracle_scaled_dice']:.3f}, "
        f"gap={analysis_row['automatic_to_oracle_gap']:.3f}, "
        f"status={analysis_row['side_status']}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def representative_figures(
    predictions: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    count: int,
) -> None:
    if count <= 0:
        return

    figure_dir = output_dir / "representative_failures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    pred = predictions.copy()
    pred["case_id"] = pred["case_id"].astype(str)
    pred["slice_idx"] = pred["slice_idx"].astype(int)
    pred_index = pred.set_index(["case_id", "slice_idx"])

    categories = [
        "absent_side_false_positive",
        "missed_side",
        "localization_or_coverage_failure",
        "box_geometry_failure",
        "oracle_limited",
        "high_gap_despite_acceptable_prompt_metrics",
        "other_low_dice",
    ]

    for category in categories:
        subset = analysis[
            (analysis["primary_category"] == category)
            & analysis["low_dice"]
        ].copy()
        if subset.empty:
            continue

        if category == "oracle_limited":
            selected = subset.nsmallest(count, "oracle_scaled_dice")
        else:
            selected = subset.nlargest(count, "positive_oracle_gap")

        category_dir = figure_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            key = (str(row["case_id"]), int(row["slice_idx"]))
            prediction_row = pred_index.loc[key]
            safe_case = str(row["case_id"]).replace("/", "_")
            output = category_dir / f"{rank:02d}_{safe_case}_z{int(row['slice_idx']):04d}.png"
            plot_prompt_overlay(prediction_row, row, output)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    prompt_details = pd.read_csv(args.prompt_details)
    medsam = pd.read_csv(args.medsam_per_slice)

    validate_inputs(predictions, prompt_details, medsam)
    analysis = build_slice_table(predictions, prompt_details, medsam)

    category_summary = grouped_summary(
        analysis,
        "primary_category",
        low_only_denominator=True,
    ).sort_values("n_slices", ascending=False)
    status_summary = grouped_summary(
        analysis,
        "side_status",
    ).sort_values("n_slices", ascending=False)
    area_summary = build_area_summary(analysis)
    case_summary = build_case_summary(analysis)
    single_gt = single_gt_comparison(analysis)

    analysis.to_csv(output_dir / "error_analysis_per_slice.csv", index=False)
    category_summary.to_csv(output_dir / "failure_category_summary.csv", index=False)
    status_summary.to_csv(output_dir / "side_status_summary.csv", index=False)
    area_summary.to_csv(output_dir / "area_error_summary.csv", index=False)
    case_summary.to_csv(output_dir / "case_error_summary.csv", index=False)
    single_gt.to_csv(output_dir / "single_gt_extra_side_by_area.csv", index=False)

    analysis[analysis["low_dice"]].sort_values(
        ["automatic_dice", "automatic_to_oracle_gap"],
        ascending=[True, False],
    ).to_csv(output_dir / "all_low_dice_slices.csv", index=False)

    analysis[analysis["large_gap_flag"]].sort_values(
        "automatic_to_oracle_gap", ascending=False
    ).to_csv(output_dir / "large_oracle_gap_slices.csv", index=False)

    analysis[
        (analysis["combined_area_bin"] == "50_90")
        & (analysis["automatic_dice"] < LOW_DICE)
        & (analysis["oracle_scaled_dice"] >= GOOD_ORACLE)
    ].sort_values(
        "automatic_to_oracle_gap", ascending=False
    ).to_csv(output_dir / "focus_50_90_good_oracle_low_auto.csv", index=False)

    analysis[
        (analysis["gt_prompt_count"] == 1)
        & (analysis["side_status"] == "extra_side")
    ].sort_values(
        ["automatic_dice", "automatic_to_oracle_gap"],
        ascending=[True, False],
    ).to_csv(output_dir / "single_gt_extra_side_slices.csv", index=False)

    summary = build_summary(
        analysis,
        case_summary,
        category_summary,
        status_summary,
        single_gt,
    )
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_histogram(analysis, output_dir / "dice_distribution.png")
    plot_auto_vs_oracle(analysis, output_dir / "automatic_vs_oracle.png")
    plot_by_area(analysis, output_dir / "dice_by_area_bin.png")
    plot_by_side_status(analysis, output_dir / "dice_by_side_status.png")
    plot_case_dice(case_summary, output_dir / "case_dice.png")

    representative_figures(
        predictions,
        analysis,
        output_dir,
        count=args.figure_count,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved descriptive error analysis to: {output_dir}")


if __name__ == "__main__":
    main()
