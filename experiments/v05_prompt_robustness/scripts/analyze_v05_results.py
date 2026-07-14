from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROMPT_METRICS = [
    "box_iou",
    "target_coverage",
    "center_error_norm",
    "box_area_ratio",
    "abs_log_area_ratio",
    "target_fill_ratio",
]

OUTCOME_METRICS = [
    "dice",
    "dice_retention",
    "delta_dice",
    "failure",
    "prompt_induced_failure",
    "relative_failure",
    "severe_drop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze V0.5 MedSAM prompt-robustness results."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the results CSV parent directory.",
    )
    parser.add_argument("--failure-dice", type=float, default=0.80)
    parser.add_argument(
        "--relative-failure-retention",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--severe-drop-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def add_or_reconstruct_metrics(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Add new metrics while remaining compatible with older result CSVs."""
    result = data.copy()

    if "abs_log_area_ratio" not in result.columns:
        safe_ratio = result["box_area_ratio"].clip(lower=1e-8)
        result["abs_log_area_ratio"] = np.abs(np.log(safe_ratio))

    if "failure" not in result.columns:
        result["failure"] = (
            result["dice"] < args.failure_dice
        ).astype(int)

    if "prompt_induced_failure" not in result.columns:
        result["prompt_induced_failure"] = (
            (result["oracle_dice"] >= args.failure_dice)
            & (result["dice"] < args.failure_dice)
        ).astype(int)

    if "relative_failure" not in result.columns:
        result["relative_failure"] = (
            result["dice_retention"]
            < args.relative_failure_retention
        ).astype(int)

    if "severe_drop" not in result.columns:
        result["severe_drop"] = (
            result["delta_dice"]
            <= -args.severe_drop_threshold
        ).astype(int)

    # New formal runs contain the true per-case maximum across all valid slices.
    # Old CSV files may not contain it, so do not fabricate it.
    if "relative_area_to_case_max" not in result.columns:
        result["relative_area_to_case_max"] = np.nan

    return result


def save_condition_table(data: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        data.groupby(
            ["split_group", "perturbation_type", "perturbation_level"],
            dropna=False,
        )
        .agg(
            evaluations=("dice", "size"),
            cases=("case_id", "nunique"),
            mean_dice=("dice", "mean"),
            std_dice=("dice", "std"),
            median_dice=("dice", "median"),
            mean_delta_dice=("delta_dice", "mean"),
            median_delta_dice=("delta_dice", "median"),
            mean_retention=("dice_retention", "mean"),
            median_retention=("dice_retention", "median"),
            raw_failure_rate=("failure", "mean"),
            prompt_induced_failure_rate=(
                "prompt_induced_failure",
                "mean",
            ),
            relative_failure_rate=("relative_failure", "mean"),
            severe_drop_rate=("severe_drop", "mean"),
            mean_box_iou=("box_iou", "mean"),
            mean_coverage=("target_coverage", "mean"),
            mean_center_error=("center_error_norm", "mean"),
            mean_area_ratio=("box_area_ratio", "mean"),
            mean_abs_log_area_ratio=("abs_log_area_ratio", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_dir / "analysis_summary_by_condition.csv",
        index=False,
    )


def spearman_long_table(
    data: pd.DataFrame,
    group_name: str,
) -> list[dict]:
    records: list[dict] = []

    for prompt_metric in PROMPT_METRICS:
        for outcome_metric in ["dice", "dice_retention", "delta_dice"]:
            subset = data[[prompt_metric, outcome_metric]].dropna()

            if (
                len(subset) < 3
                or subset[prompt_metric].nunique() < 2
                or subset[outcome_metric].nunique() < 2
            ):
                correlation = np.nan
            else:
                correlation = subset.corr(
                    method="spearman"
                ).iloc[0, 1]

            records.append(
                {
                    "group": group_name,
                    "prompt_metric": prompt_metric,
                    "outcome_metric": outcome_metric,
                    "spearman_rho": correlation,
                    "evaluations": len(subset),
                }
            )

    return records


def save_correlations(data: pd.DataFrame, output_dir: Path) -> None:
    non_oracle = data[data["perturbation_type"] != "oracle"].copy()

    available_metrics = [
        column
        for column in PROMPT_METRICS + ["dice", "dice_retention"]
        if column in non_oracle.columns
    ]
    overall_matrix = non_oracle[available_metrics].corr(method="spearman")
    overall_matrix.to_csv(output_dir / "spearman_correlations.csv")

    records = spearman_long_table(non_oracle, "all_non_oracle")

    for perturbation_type, subset in non_oracle.groupby(
        "perturbation_type",
        sort=True,
    ):
        records.extend(
            spearman_long_table(
                subset,
                str(perturbation_type),
            )
        )

    long_table = pd.DataFrame(records)
    long_table.to_csv(
        output_dir / "spearman_by_perturbation.csv",
        index=False,
    )


def case_bootstrap_summary(
    group: pd.DataFrame,
    metrics: list[str],
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap case-level means, not individual repeated prompts."""
    case_means = (
        group.groupby("case_id", sort=False)[metrics]
        .mean()
        .dropna(how="all")
    )

    result: dict[str, float | int] = {
        "cases": int(len(case_means)),
        "evaluations": int(len(group)),
    }
    if case_means.empty:
        return result

    rng = np.random.default_rng(seed)
    number_of_cases = len(case_means)
    values = case_means.to_numpy(dtype=float)

    bootstrap_means = np.empty(
        (repeats, len(metrics)),
        dtype=float,
    )
    for repeat_index in range(repeats):
        sampled_indices = rng.integers(
            0,
            number_of_cases,
            size=number_of_cases,
        )
        bootstrap_means[repeat_index] = np.nanmean(
            values[sampled_indices],
            axis=0,
        )

    for metric_index, metric in enumerate(metrics):
        original = float(np.nanmean(values[:, metric_index]))
        lower, upper = np.nanpercentile(
            bootstrap_means[:, metric_index],
            [2.5, 97.5],
        )
        result[f"{metric}_case_mean"] = original
        result[f"{metric}_ci95_low"] = float(lower)
        result[f"{metric}_ci95_high"] = float(upper)

    return result


def save_bootstrap_confidence_intervals(
    data: pd.DataFrame,
    output_dir: Path,
    repeats: int,
    seed: int,
) -> None:
    metrics = [
        "dice",
        "delta_dice",
        "dice_retention",
        "prompt_induced_failure",
        "relative_failure",
        "severe_drop",
        "box_iou",
        "target_coverage",
    ]

    records = []
    grouped = data.groupby(
        ["split_group", "perturbation_type", "perturbation_level"],
        dropna=False,
        sort=True,
    )

    for group_index, (keys, group) in enumerate(grouped):
        split_group, perturbation_type, perturbation_level = keys
        record = {
            "split_group": split_group,
            "perturbation_type": perturbation_type,
            "perturbation_level": perturbation_level,
        }
        record.update(
            case_bootstrap_summary(
                group=group,
                metrics=metrics,
                repeats=repeats,
                seed=seed + group_index,
            )
        )
        records.append(record)

    pd.DataFrame(records).to_csv(
        output_dir / "case_bootstrap_ci.csv",
        index=False,
    )


def threshold_rows(
    data: pd.DataFrame,
    metric: str,
    thresholds: list[float],
) -> pd.DataFrame:
    records = []
    non_oracle = data[data["perturbation_type"] != "oracle"].copy()

    for threshold in thresholds:
        subset = non_oracle[non_oracle[metric] >= threshold]
        if subset.empty:
            continue

        records.append(
            {
                "metric": metric,
                "threshold": threshold,
                "evaluations": len(subset),
                "cases": subset["case_id"].nunique(),
                "median_dice": subset["dice"].median(),
                "mean_dice": subset["dice"].mean(),
                "median_retention": subset["dice_retention"].median(),
                "mean_retention": subset["dice_retention"].mean(),
                "raw_failure_rate": subset["failure"].mean(),
                "prompt_induced_failure_rate": (
                    subset["prompt_induced_failure"].mean()
                ),
                "relative_failure_rate": (
                    subset["relative_failure"].mean()
                ),
                "severe_drop_rate": subset["severe_drop"].mean(),
            }
        )

    return pd.DataFrame(records)


def save_threshold_analysis(data: pd.DataFrame, output_dir: Path) -> None:
    tables = [
        threshold_rows(
            data,
            "box_iou",
            [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90],
        ),
        threshold_rows(
            data,
            "target_coverage",
            [0.50, 0.75, 0.85, 0.90, 0.95, 0.98, 0.99],
        ),
    ]
    result = pd.concat(tables, ignore_index=True)
    result["acceptable"] = (
        (result["median_retention"] >= 0.95)
        & (result["prompt_induced_failure_rate"] <= 0.05)
        & (result["severe_drop_rate"] <= 0.05)
    )
    result.to_csv(
        output_dir / "threshold_analysis.csv",
        index=False,
    )


def save_size_group_analysis(data: pd.DataFrame, output_dir: Path) -> None:
    oracle = (
        data[data["perturbation_type"] == "oracle"]
        .drop_duplicates(["case_id", "slice_index", "target_name"])
    )

    columns = ["mask_area_group"]
    if oracle["relative_area_to_case_max"].notna().any():
        columns.append("relative_area_to_case_max")

    summary = (
        oracle.groupby("mask_area_group", dropna=False)
        .agg(
            samples=("dice", "size"),
            cases=("case_id", "nunique"),
            mean_mask_area=("mask_area", "mean"),
            median_mask_area=("mask_area", "median"),
            mean_relative_area=(
                "relative_area_to_case_max",
                "mean",
            ),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            std_dice=("dice", "std"),
            min_dice=("dice", "min"),
            max_dice=("dice", "max"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_dir / "oracle_by_target_size.csv",
        index=False,
    )


def plot_curve(
    data: pd.DataFrame,
    perturbation_type: str,
    output_path: Path,
) -> None:
    subset = data[data["perturbation_type"] == perturbation_type].copy()
    if subset.empty:
        return

    grouped = (
        subset.groupby("perturbation_level")["dice"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .sort_values("perturbation_level")
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(
        grouped["perturbation_level"],
        grouped["mean"],
        yerr=grouped["std"].fillna(0.0),
        marker="o",
        capsize=3,
        label="Mean ± SD",
    )
    axis.plot(
        grouped["perturbation_level"],
        grouped["median"],
        marker="s",
        label="Median",
    )
    axis.set_xlabel("Perturbation level")
    axis.set_ylabel("Dice")
    axis.set_title(f"Dice vs {perturbation_type}")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_scatter(
    data: pd.DataFrame,
    x_column: str,
    output_path: Path,
) -> None:
    subset = data[data["perturbation_type"] != "oracle"].copy()
    if subset.empty:
        return

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(subset[x_column], subset["dice"], s=9, alpha=0.25)
    axis.set_xlabel(x_column)
    axis.set_ylabel("Dice")
    axis.set_title(f"{x_column} vs Dice")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_area_groups(data: pd.DataFrame, output_path: Path) -> None:
    order = ["small", "medium", "large"]
    values = [
        data.loc[
            (data["mask_area_group"] == group)
            & (data["perturbation_type"] != "oracle"),
            "dice_retention",
        ].dropna().to_numpy()
        for group in order
    ]
    if not any(len(value) for value in values):
        return

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, tick_labels=order, showfliers=False)
    axis.axhline(0.95, linestyle="--", linewidth=1)
    axis.set_xlabel("Target area group")
    axis.set_ylabel("Dice retention")
    axis.set_title("Prompt robustness by target size")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    output_dir = args.output_dir or args.results.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    data = pd.read_csv(args.results)
    required = {
        "case_id",
        "slice_index",
        "target_name",
        "split_group",
        "perturbation_type",
        "perturbation_level",
        "box_iou",
        "target_coverage",
        "center_error_norm",
        "box_area_ratio",
        "target_fill_ratio",
        "oracle_dice",
        "dice",
        "delta_dice",
        "dice_retention",
        "mask_area",
        "mask_area_group",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Results CSV is missing columns: {sorted(missing)}"
        )

    data = add_or_reconstruct_metrics(data, args)

    save_condition_table(data, output_dir)
    save_correlations(data, output_dir)
    save_bootstrap_confidence_intervals(
        data,
        output_dir,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    save_threshold_analysis(data, output_dir)
    save_size_group_analysis(data, output_dir)

    for perturbation_type in ["scale", "shift", "corner_jitter"]:
        plot_curve(
            data,
            perturbation_type,
            figures_dir / f"dice_vs_{perturbation_type}.png",
        )

    for x_column in [
        "box_iou",
        "target_coverage",
        "center_error_norm",
        "box_area_ratio",
        "abs_log_area_ratio",
    ]:
        plot_scatter(
            data,
            x_column,
            figures_dir / f"{x_column}_vs_dice.png",
        )

    plot_area_groups(
        data,
        figures_dir / "dice_retention_by_target_size.png",
    )

    non_oracle = data[data["perturbation_type"] != "oracle"]
    correlations = (
        non_oracle[
            [
                "box_iou",
                "target_coverage",
                "center_error_norm",
                "box_area_ratio",
                "abs_log_area_ratio",
                "dice",
            ]
        ]
        .corr(method="spearman")["dice"]
        .drop("dice")
        .sort_values(ascending=False)
    )

    report = {
        "evaluations": int(len(data)),
        "cases": int(data["case_id"].nunique()),
        "slice_target_samples": int(
            len(
                data[
                    ["case_id", "slice_index", "target_name"]
                ].drop_duplicates()
            )
        ),
        "mean_oracle_dice": float(
            data.loc[
                data["perturbation_type"] == "oracle",
                "dice",
            ].mean()
        ),
        "failure_definitions": {
            "raw_failure": f"dice < {args.failure_dice}",
            "prompt_induced_failure": (
                f"oracle_dice >= {args.failure_dice} and "
                f"dice < {args.failure_dice}"
            ),
            "relative_failure": (
                f"dice_retention < {args.relative_failure_retention}"
            ),
            "severe_drop": (
                f"delta_dice <= -{args.severe_drop_threshold}"
            ),
        },
        "spearman_with_dice": {
            name: (
                None if pd.isna(value) else float(value)
            )
            for name, value in correlations.items()
        },
    }
    with (output_dir / "analysis_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Analysis complete.")
    print("Evaluations:", report["evaluations"])
    print("Cases:", report["cases"])
    print("Mean oracle Dice:", f"{report['mean_oracle_dice']:.4f}")
    print("\nSpearman correlation with Dice:")
    print(correlations)
    print("\nNew outputs:")
    print(" - spearman_by_perturbation.csv")
    print(" - case_bootstrap_ci.csv")
    print(" - oracle_by_target_size.csv")
    print(" - threshold_analysis.csv")
    print("\nOutput:", output_dir)


if __name__ == "__main__":
    main()
