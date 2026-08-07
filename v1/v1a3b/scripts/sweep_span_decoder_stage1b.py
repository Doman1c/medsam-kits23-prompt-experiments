from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a3b.span_decoder import (
    SIDES,
    SpanDecoderConfig,
    apply_span_decoder,
    summarize_presence_topology,
)
from v1a1.calibration import add_threshold_independent_smoothing


def parse_float_grid(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("Empty float grid")
    return values


def parse_int_grid(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("Empty integer grid")
    return values


def validate_validation_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    frame_keys = set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))
    val_keys = set(zip(val["case_id"].astype(str), val["slice_idx"].astype(int)))
    if frame_keys != val_keys:
        raise RuntimeError("Prediction rows do not exactly match validation split; refusing to sweep")
    if int(val["case_id"].astype(str).nunique()) != 20 or len(val) != 3644:
        raise RuntimeError(
            f"Unexpected validation split: cases={val['case_id'].nunique()}, rows={len(val)}"
        )


def pad_single_span_predictions(frame: pd.DataFrame, boundary_pad: int) -> pd.DataFrame:
    """Expand the one decoded span by a fixed number of z-slices on each side.

    This operation uses only predicted span boundaries. It never accesses GT labels.
    """
    pad = int(boundary_pad)
    if pad < 0:
        raise ValueError("boundary_pad must be non-negative")
    if pad == 0:
        return frame.copy()

    output = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    for _, indices in output.groupby("case_id", sort=False).groups.items():
        ordered = output.loc[list(indices)].sort_values("slice_idx")
        ordered_indices = ordered.index.to_numpy(dtype=int)
        for side in SIDES:
            present = ordered[f"{side}_smooth_presence"].to_numpy(dtype=np.uint8).astype(bool)
            positive = np.flatnonzero(present)
            if not len(positive):
                continue
            start = max(int(positive[0]) - pad, 0)
            end = min(int(positive[-1]) + pad, len(present) - 1)
            padded = np.zeros(len(present), dtype=np.uint8)
            padded[start : end + 1] = 1
            output.loc[ordered_indices, f"{side}_smooth_presence"] = padded
            output.loc[ordered_indices, f"{side}_span_presence"] = padded

    for side in SIDES:
        output[f"{side}_smooth_presence"] = output[f"{side}_smooth_presence"].astype(np.uint8)
        output[f"{side}_span_presence"] = output[f"{side}_span_presence"].astype(np.uint8)
    return output


def decode_with_padding(
    frame: pd.DataFrame,
    config: SpanDecoderConfig,
    boundary_pad: int,
) -> pd.DataFrame:
    decoded = apply_span_decoder(frame, config)
    return pad_single_span_predictions(decoded, boundary_pad=boundary_pad)


def metrics_row(config: SpanDecoderConfig, boundary_pad: int, summary: dict) -> dict:
    return {
        **config.to_dict(),
        "boundary_pad": int(boundary_pad),
        "positive_recall": summary["positive_recall"],
        "left_positive_recall": summary["left_positive_recall"],
        "right_positive_recall": summary["right_positive_recall"],
        "false_positive_rate": summary["false_positive_rate"],
        "specificity": summary["specificity"],
        "left_false_positive_rate": summary["left_false_positive_rate"],
        "right_false_positive_rate": summary["right_false_positive_rate"],
        "multi_segment_case_side_rate": summary["multi_segment_case_side_rate"],
        "false_positive_segment_count": summary["false_positive_segment_count"],
        "false_positive_segment_slices": summary["false_positive_segment_slices"],
        "onset_abs_error_mean": summary["onset_abs_error_mean"],
        "offset_abs_error_mean": summary["offset_abs_error_mean"],
    }


def config_key(row: pd.Series) -> tuple:
    return (
        str(row["mode"]),
        float(row["left_threshold"]),
        float(row["right_threshold"]),
        int(row["baseline_max_gap"]),
        int(row["max_internal_gap"]),
        int(row["min_span_length"]),
        float(row["span_penalty"]),
        float(row["length_penalty"]),
        int(row["boundary_pad"]),
    )


def config_from_row(row: pd.Series) -> tuple[SpanDecoderConfig, int]:
    config = SpanDecoderConfig(
        mode=str(row["mode"]),
        left_threshold=float(row["left_threshold"]),
        right_threshold=float(row["right_threshold"]),
        baseline_max_gap=int(row["baseline_max_gap"]),
        max_internal_gap=int(row["max_internal_gap"]),
        min_span_length=int(row["min_span_length"]),
        span_penalty=float(row["span_penalty"]),
        length_penalty=float(row["length_penalty"]),
    )
    return config, int(row["boundary_pad"])


def choose_diverse_rows(feasible: pd.DataFrame, limit: int) -> list[pd.Series]:
    if feasible.empty or limit <= 0:
        return []
    feasible = feasible.copy()
    feasible["boundary_mae"] = feasible[
        ["onset_abs_error_mean", "offset_abs_error_mean"]
    ].mean(axis=1)

    picks: list[pd.Series] = []
    seen: set[tuple] = set()

    orderings = [
        (
            ["false_positive_rate", "positive_recall", "boundary_mae"],
            [True, False, True],
        ),
        (
            ["positive_recall", "false_positive_rate", "boundary_mae"],
            [False, True, True],
        ),
        (
            ["boundary_mae", "false_positive_rate", "positive_recall"],
            [True, True, False],
        ),
    ]
    for columns, ascending in orderings:
        ordered = feasible.sort_values(columns, ascending=ascending)
        for _, row in ordered.iterrows():
            key = config_key(row)
            if key in seen:
                continue
            seen.add(key)
            picks.append(row)
            break
        if len(picks) >= limit:
            return picks

    if len(picks) < limit:
        ordered = feasible.sort_values(
            ["false_positive_rate", "positive_recall"],
            ascending=[True, False],
        )
        for _, row in ordered.iterrows():
            key = config_key(row)
            if key in seen:
                continue
            seen.add(key)
            picks.append(row)
            if len(picks) >= limit:
                break
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1b validation sweep for V1a.3-B. Extends the original "
            "single-span/optimal-span grid with lower thresholds, wider internal "
            "gaps, and GT-independent z-boundary padding."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--left-thresholds",
        default="0.0005,0.001,0.0015,0.002,0.0025,0.0035,0.005",
    )
    parser.add_argument(
        "--right-thresholds",
        default="0.001,0.0025,0.005,0.0075,0.01,0.02,0.04",
    )
    parser.add_argument("--single-max-gaps", default="2,3,4,5")
    parser.add_argument("--optimal-max-internal-gaps", default="2,3,4,5,6")
    parser.add_argument("--boundary-pads", default="0,1,2,3,4,6,8")
    parser.add_argument("--min-overall-recall", type=float, default=0.95)
    parser.add_argument("--min-side-recall", type=float, default=0.90)
    parser.add_argument("--shortlist-per-mode", type=int, default=3)
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    frame = pd.read_csv(prediction_path, dtype={"case_id": str})
    validate_validation_rows(frame, manifest_path)
    frame = add_threshold_independent_smoothing(frame)

    left_thresholds = parse_float_grid(args.left_thresholds)
    right_thresholds = parse_float_grid(args.right_thresholds)
    single_gaps = parse_int_grid(args.single_max_gaps)
    optimal_gaps = parse_int_grid(args.optimal_max_internal_gaps)
    boundary_pads = parse_int_grid(args.boundary_pads)

    jobs: list[tuple[SpanDecoderConfig, int]] = []
    for left_threshold, right_threshold, gap, pad in itertools.product(
        left_thresholds,
        right_thresholds,
        single_gaps,
        boundary_pads,
    ):
        jobs.append(
            (
                SpanDecoderConfig(
                    mode="single_span",
                    left_threshold=left_threshold,
                    right_threshold=right_threshold,
                    baseline_max_gap=gap,
                    min_span_length=1,
                ),
                pad,
            )
        )

    for left_threshold, right_threshold, gap, pad in itertools.product(
        left_thresholds,
        right_thresholds,
        optimal_gaps,
        boundary_pads,
    ):
        jobs.append(
            (
                SpanDecoderConfig(
                    mode="optimal_span",
                    left_threshold=left_threshold,
                    right_threshold=right_threshold,
                    max_internal_gap=gap,
                    min_span_length=1,
                    span_penalty=0.0,
                    length_penalty=0.0,
                ),
                pad,
            )
        )

    rows: list[dict] = []
    for i, (config, pad) in enumerate(jobs, start=1):
        decoded = decode_with_padding(frame, config, boundary_pad=pad)
        summary, _, _ = summarize_presence_topology(decoded)
        row = metrics_row(config, pad, summary)
        row["constraints_met"] = bool(
            row["positive_recall"] >= args.min_overall_recall
            and row["left_positive_recall"] >= args.min_side_recall
            and row["right_positive_recall"] >= args.min_side_recall
        )
        rows.append(row)
        if i % 100 == 0 or i == len(jobs):
            print(f"Swept {i}/{len(jobs)} stage-1b configs")

    sweep = pd.DataFrame(rows).sort_values(
        ["constraints_met", "false_positive_rate", "positive_recall"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    baseline_config = SpanDecoderConfig(
        mode="baseline",
        left_threshold=0.005,
        right_threshold=0.005,
        baseline_max_gap=2,
    )
    baseline_decoded = apply_span_decoder(frame, baseline_config)
    baseline_summary, _, _ = summarize_presence_topology(baseline_decoded)
    baseline_row = metrics_row(baseline_config, 0, baseline_summary)
    baseline_row["constraints_met"] = bool(
        baseline_row["positive_recall"] >= args.min_overall_recall
        and baseline_row["left_positive_recall"] >= args.min_side_recall
        and baseline_row["right_positive_recall"] >= args.min_side_recall
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "span_decoder_stage1b_sweep.csv", index=False)
    pd.DataFrame([baseline_row]).to_csv(
        output_dir / "b0_locked_baseline_metrics.csv",
        index=False,
    )

    candidates = []
    baseline_dir = output_dir / "candidates" / "b0_locked_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_decoded.to_csv(baseline_dir / "recommended_predictions.csv", index=False)
    _, baseline_case_side, baseline_false = summarize_presence_topology(baseline_decoded)
    baseline_case_side.to_csv(baseline_dir / "topology_case_side.csv", index=False)
    baseline_false.to_csv(baseline_dir / "false_positive_segments.csv", index=False)

    selected_specs: list[tuple[str, SpanDecoderConfig, int]] = [
        ("b0_locked_baseline", baseline_config, 0)
    ]
    for mode, prefix in (
        ("single_span", "b1_single_span"),
        ("optimal_span", "b2_optimal_span"),
    ):
        feasible = sweep[(sweep["mode"] == mode) & sweep["constraints_met"]].copy()
        chosen = choose_diverse_rows(feasible, limit=args.shortlist_per_mode)
        for rank, row in enumerate(chosen, start=1):
            config, pad = config_from_row(row)
            selected_specs.append((f"{prefix}_{rank}", config, pad))

    for name, config, pad in selected_specs:
        candidate_dir = output_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if name == "b0_locked_baseline":
            decoded = baseline_decoded
            summary = baseline_summary
            case_side = baseline_case_side
            false_segments = baseline_false
        else:
            decoded = decode_with_padding(frame, config, boundary_pad=pad)
            summary, case_side, false_segments = summarize_presence_topology(decoded)
            decoded.to_csv(candidate_dir / "recommended_predictions.csv", index=False)
            case_side.to_csv(candidate_dir / "topology_case_side.csv", index=False)
            false_segments.to_csv(candidate_dir / "false_positive_segments.csv", index=False)

        decoder = {**config.to_dict(), "boundary_pad": int(pad)}
        payload = {
            "status": "validation_only_v1a3b_stage1b_decoder_candidate",
            "test_set_used": False,
            "name": name,
            "decoder": decoder,
            "presence_metrics": summary,
            "constraints": {
                "min_overall_recall": args.min_overall_recall,
                "min_side_recall": args.min_side_recall,
            },
            "constraints_met": bool(
                summary["positive_recall"] >= args.min_overall_recall
                and summary["left_positive_recall"] >= args.min_side_recall
                and summary["right_positive_recall"] >= args.min_side_recall
            ),
        }
        (candidate_dir / "decoder_config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        calibration = {
            "configuration": {
                "variant": "smooth",
                "max_gap": 0,
                "left_threshold": config.left_threshold,
                "right_threshold": config.right_threshold,
            },
            "decoder": decoder,
            "status": "v1a3b_stage1b_materialized_predictions",
            "test_set_used": False,
        }
        calibration_path = candidate_dir / "recommended_calibration.json"
        calibration_path.write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        candidates.append(
            {
                "name": name,
                "directory": str(candidate_dir),
                "predictions": str(candidate_dir / "recommended_predictions.csv"),
                "calibration_json": str(calibration_path),
                "expansion_scale": float(args.expansion_scale),
                "decoder": decoder,
                "presence_metrics": summary,
            }
        )

    manifest_payload = {
        "status": "validation_only_v1a3b_stage1b_shortlist",
        "test_set_used": False,
        "source_predictions": str(prediction_path),
        "selection": (
            "B0 is the frozen V1a.3-A baseline. B1/B2 must satisfy overall recall "
            ">= 0.95 and left/right recall >= 0.90. Within each decoder mode retain "
            "low-FPR, high-recall, and low-boundary-error representatives."
        ),
        "candidates": candidates,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_payload = {
        "status": "validation_only_v1a3b_stage1b_sweep",
        "test_set_used": False,
        "reason_for_stage1b": (
            "Stage 1 produced zero feasible B1/B2 configs. Best recall occurred "
            "at the lowest tested threshold and largest tested gap, while B2 "
            "substantially reduced FPR and boundary MAE. Stage 1b widens only "
            "these boundary-of-grid dimensions and adds GT-independent span padding."
        ),
        "num_swept_configs": int(len(sweep)),
        "num_feasible_configs": int(sweep["constraints_met"].sum()),
        "feasible_by_mode": {
            mode: int(
                ((sweep["mode"] == mode) & sweep["constraints_met"]).sum()
            )
            for mode in ("single_span", "optimal_span")
        },
        "baseline": baseline_row,
        "shortlist": [item["name"] for item in candidates],
    }
    (output_dir / "sweep_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
