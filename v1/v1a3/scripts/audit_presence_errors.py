from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIDES = ("left", "right")


def nearest_distance(slices: np.ndarray, positive: np.ndarray) -> np.ndarray:
    if len(positive) == 0:
        return np.full(len(slices), np.inf, dtype=np.float64)
    return np.min(np.abs(slices[:, None] - positive[None, :]), axis=1)


def span_position(slice_index: int, positive: np.ndarray) -> str:
    if len(positive) == 0:
        return "case_has_no_positive_side"
    if slice_index < int(positive.min()):
        return "before_span"
    if slice_index > int(positive.max()):
        return "after_span"
    return "inside_span"


def negative_subtype(
    *,
    other_side_present: bool,
    position: str,
    distance: float,
) -> str:
    if other_side_present:
        return "contralateral_negative"
    if position == "inside_span":
        return "inside_span_negative"
    if distance <= 3:
        return "boundary_01_03"
    if distance <= 8:
        return "boundary_04_08"
    if np.isfinite(distance):
        return "far_negative"
    return "case_without_side_positive"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit presence errors and export row-level hard negatives"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--variant", choices=["raw", "smooth"], default="smooth")
    parser.add_argument("--left-threshold", type=float, default=0.20)
    parser.add_argument("--right-threshold", type=float, default=0.30)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--hard-negative-min-score", type=float, default=0.20)
    parser.add_argument("--hard-negative-top-k-per-case", type=int, default=32)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root / "v1" / "v1a2b"))
    from v1a2b.calibration import (
        CalibrationConfig,
        add_threshold_independent_smoothing,
        apply_calibration,
        load_config,
    )

    manifest = pd.read_csv(args.manifest)
    manifest["case_id"] = manifest["case_id"].astype(str)
    manifest["slice_idx"] = manifest["slice_idx"].astype(int)
    manifest = manifest[manifest["split"].astype(str) == args.split].copy()
    if len(manifest) == 0:
        raise RuntimeError(f"No manifest rows for split={args.split!r}")

    predictions = pd.read_csv(args.predictions)
    predictions["case_id"] = predictions["case_id"].astype(str)
    predictions["slice_idx"] = predictions["slice_idx"].astype(int)

    config = (
        load_config(args.calibration_json)
        if args.calibration_json
        else CalibrationConfig(
            variant=args.variant,
            max_gap=args.max_gap,
            left_threshold=args.left_threshold,
            right_threshold=args.right_threshold,
        )
    )
    calibrated = add_threshold_independent_smoothing(predictions)
    calibrated = apply_calibration(calibrated, config)

    merged = calibrated.merge(
        manifest[
            [
                "case_id",
                "slice_idx",
                "split",
                "left_presence",
                "right_presence",
            ]
        ],
        on=["case_id", "slice_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(manifest):
        raise RuntimeError(
            f"Prediction/manifest mismatch: matched={len(merged)}, "
            f"manifest={len(manifest)}"
        )

    records = []
    for case_id, case in merged.groupby("case_id", sort=False):
        case = case.sort_values("slice_idx")
        slices = case["slice_idx"].to_numpy(dtype=int)
        positive_slices = {
            side: slices[case[f"{side}_presence"].to_numpy(dtype=int) == 1]
            for side in SIDES
        }
        for side in SIDES:
            other = "right" if side == "left" else "left"
            distance = nearest_distance(slices, positive_slices[side])
            for local_index, (_, row) in enumerate(case.iterrows()):
                gt = int(row[f"{side}_presence"])
                other_gt = int(row[f"{other}_presence"])
                probability = float(
                    row[f"{side}_{config.variant}_presence_probability"]
                )
                pred = int(row[f"{side}_{config.variant}_presence"])
                position = span_position(
                    int(row["slice_idx"]),
                    positive_slices[side],
                )
                if gt:
                    subtype = "positive"
                    outcome = "tp" if pred else "fn"
                else:
                    subtype = negative_subtype(
                        other_side_present=bool(other_gt),
                        position=position,
                        distance=float(distance[local_index]),
                    )
                    outcome = "fp" if pred else "tn"

                records.append(
                    {
                        "case_id": case_id,
                        "slice_idx": int(row["slice_idx"]),
                        "split": args.split,
                        "side": side,
                        "gt_presence": gt,
                        "other_side_gt_presence": other_gt,
                        "presence_probability": probability,
                        "pred_presence": pred,
                        "outcome": outcome,
                        "negative_subtype": subtype,
                        "position_relative_to_span": position,
                        "distance_to_nearest_positive_slice": (
                            float(distance[local_index])
                            if np.isfinite(distance[local_index])
                            else np.nan
                        ),
                    }
                )

    details = pd.DataFrame(records)
    positive = details[details["gt_presence"] == 1].copy()
    negative = details[details["gt_presence"] == 0].copy()

    subtype_rows = []
    for (side, subtype), subset in negative.groupby(
        ["side", "negative_subtype"],
        sort=True,
        dropna=False,
    ):
        subtype_rows.append(
            {
                "side": side,
                "negative_subtype": subtype,
                "n": int(len(subset)),
                "false_positive_count": int((subset["outcome"] == "fp").sum()),
                "false_positive_rate": float((subset["outcome"] == "fp").mean()),
                "probability_mean": float(subset["presence_probability"].mean()),
                "probability_median": float(
                    subset["presence_probability"].median()
                ),
                "probability_p90": float(
                    subset["presence_probability"].quantile(0.90)
                ),
            }
        )
    subtype_summary = pd.DataFrame(subtype_rows)

    row_pool = (
        negative.groupby(["case_id", "slice_idx", "split"], as_index=False)
        .agg(
            hard_negative_score=("presence_probability", "max"),
            any_false_positive=(
                "outcome",
                lambda values: int((values == "fp").any()),
            ),
        )
    )
    row_pool = row_pool[
        (row_pool["any_false_positive"] == 1)
        | (
            row_pool["hard_negative_score"]
            >= float(args.hard_negative_min_score)
        )
    ].copy()
    row_pool = row_pool.sort_values(
        ["case_id", "hard_negative_score", "slice_idx"],
        ascending=[True, False, True],
    )
    if args.hard_negative_top_k_per_case > 0:
        row_pool = row_pool.groupby("case_id", sort=False).head(
            args.hard_negative_top_k_per_case
        )

    row_pool["left_raw_presence_probability"] = row_pool[
        "hard_negative_score"
    ]
    row_pool["right_raw_presence_probability"] = row_pool[
        "hard_negative_score"
    ]
    row_pool["left_presence"] = 0
    row_pool["right_presence"] = 0

    summary = {
        "split": args.split,
        "rows": int(len(merged)),
        "positive_side_samples": int(len(positive)),
        "negative_side_samples": int(len(negative)),
        "positive_recall": float((positive["outcome"] == "tp").mean()),
        "false_positive_rate": float((negative["outcome"] == "fp").mean()),
        "left_recall": float(
            (
                positive.loc[positive["side"] == "left", "outcome"]
                == "tp"
            ).mean()
        ),
        "right_recall": float(
            (
                positive.loc[positive["side"] == "right", "outcome"]
                == "tp"
            ).mean()
        ),
        "hard_negative_pool_rows": int(len(row_pool)),
        "hard_negative_pool_cases": int(row_pool["case_id"].nunique())
        if len(row_pool)
        else 0,
        "configuration": config.to_dict(),
        "negative_subtypes": subtype_summary.to_dict(orient="records"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "presence_error_details.csv", index=False)
    subtype_summary.to_csv(
        output_dir / "presence_error_by_subtype.csv",
        index=False,
    )
    row_pool.to_csv(output_dir / "hard_negative_pool.csv", index=False)
    (output_dir / "presence_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
