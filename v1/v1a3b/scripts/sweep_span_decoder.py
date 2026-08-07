from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a3b.span_decoder import SpanDecoderConfig, apply_span_decoder, summarize_presence_topology
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


def metrics_row(config: SpanDecoderConfig, summary: dict) -> dict:
    return {
        **config.to_dict(),
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
        row["mode"],
        float(row["left_threshold"]),
        float(row["right_threshold"]),
        int(row["baseline_max_gap"]),
        int(row["max_internal_gap"]),
        int(row["min_span_length"]),
        float(row["span_penalty"]),
        float(row["length_penalty"]),
    )


def config_from_row(row: pd.Series) -> SpanDecoderConfig:
    return SpanDecoderConfig(
        mode=str(row["mode"]),
        left_threshold=float(row["left_threshold"]),
        right_threshold=float(row["right_threshold"]),
        baseline_max_gap=int(row["baseline_max_gap"]),
        max_internal_gap=int(row["max_internal_gap"]),
        min_span_length=int(row["min_span_length"]),
        span_penalty=float(row["span_penalty"]),
        length_penalty=float(row["length_penalty"]),
    )


def select_shortlist(
    sweep: pd.DataFrame,
    baseline_config: SpanDecoderConfig,
    per_mode: int,
) -> list[tuple[str, SpanDecoderConfig]]:
    selected: list[tuple[str, SpanDecoderConfig]] = [("b0_locked_baseline", baseline_config)]
    for mode, prefix in (("single_span", "b1_single_span"), ("optimal_span", "b2_optimal_span")):
        feasible = sweep[(sweep["mode"] == mode) & (sweep["constraints_met"])].copy()
        if feasible.empty:
            continue
        feasible["boundary_mae"] = feasible[["onset_abs_error_mean", "offset_abs_error_mean"]].mean(axis=1)
        feasible = feasible.sort_values(
            ["false_positive_rate", "positive_recall", "boundary_mae", "false_positive_segment_count"],
            ascending=[True, False, True, True],
        )
        used = set()
        rank = 0
        for _, row in feasible.iterrows():
            key = config_key(row)
            if key in used:
                continue
            used.add(key)
            rank += 1
            selected.append((f"{prefix}_{rank}", config_from_row(row)))
            if rank >= per_mode:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only sweep of V1a.3-B case-level span decoders")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.0025,0.005,0.0075,0.01,0.02,0.04,0.08,0.10,0.15")
    parser.add_argument("--baseline-max-gaps", default="0,1,2")
    parser.add_argument("--single-min-span-lengths", default="1,3")
    parser.add_argument("--optimal-max-internal-gaps", default="1,2,3")
    parser.add_argument("--optimal-min-span-lengths", default="1,3")
    parser.add_argument("--optimal-span-penalties", default="0,1")
    parser.add_argument("--optimal-length-penalties", default="0")
    parser.add_argument("--min-overall-recall", type=float, default=0.95)
    parser.add_argument("--min-side-recall", type=float, default=0.90)
    parser.add_argument("--shortlist-per-mode", type=int, default=2)
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    frame = pd.read_csv(prediction_path, dtype={"case_id": str})
    validate_validation_rows(frame, manifest_path)
    frame = add_threshold_independent_smoothing(frame)

    thresholds = parse_float_grid(args.thresholds)
    baseline_gaps = parse_int_grid(args.baseline_max_gaps)
    single_min_lengths = parse_int_grid(args.single_min_span_lengths)
    optimal_gaps = parse_int_grid(args.optimal_max_internal_gaps)
    optimal_min_lengths = parse_int_grid(args.optimal_min_span_lengths)
    span_penalties = parse_float_grid(args.optimal_span_penalties)
    length_penalties = parse_float_grid(args.optimal_length_penalties)

    configs: list[SpanDecoderConfig] = []
    for left_threshold, right_threshold, gap, min_length in itertools.product(
        thresholds, thresholds, baseline_gaps, single_min_lengths
    ):
        configs.append(
            SpanDecoderConfig(
                mode="single_span",
                left_threshold=left_threshold,
                right_threshold=right_threshold,
                baseline_max_gap=gap,
                min_span_length=min_length,
            )
        )
    for left_threshold, right_threshold, max_internal_gap, min_length, span_penalty, length_penalty in itertools.product(
        thresholds,
        thresholds,
        optimal_gaps,
        optimal_min_lengths,
        span_penalties,
        length_penalties,
    ):
        configs.append(
            SpanDecoderConfig(
                mode="optimal_span",
                left_threshold=left_threshold,
                right_threshold=right_threshold,
                max_internal_gap=max_internal_gap,
                min_span_length=min_length,
                span_penalty=span_penalty,
                length_penalty=length_penalty,
            )
        )

    rows: list[dict] = []
    for i, config in enumerate(configs, start=1):
        decoded = apply_span_decoder(frame, config)
        summary, _, _ = summarize_presence_topology(decoded)
        row = metrics_row(config, summary)
        row["constraints_met"] = bool(
            row["positive_recall"] >= args.min_overall_recall
            and row["left_positive_recall"] >= args.min_side_recall
            and row["right_positive_recall"] >= args.min_side_recall
        )
        rows.append(row)
        if i % 100 == 0 or i == len(configs):
            print(f"Swept {i}/{len(configs)} configs")

    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values(
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
    baseline_row = metrics_row(baseline_config, baseline_summary)
    baseline_row["constraints_met"] = bool(
        baseline_row["positive_recall"] >= args.min_overall_recall
        and baseline_row["left_positive_recall"] >= args.min_side_recall
        and baseline_row["right_positive_recall"] >= args.min_side_recall
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "span_decoder_sweep.csv", index=False)
    pd.DataFrame([baseline_row]).to_csv(output_dir / "b0_locked_baseline_metrics.csv", index=False)

    shortlist = select_shortlist(sweep, baseline_config, per_mode=args.shortlist_per_mode)
    candidates = []
    for name, config in shortlist:
        candidate_dir = output_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        decoded = apply_span_decoder(frame, config)
        summary, case_side, false_segments = summarize_presence_topology(decoded)
        decoded.to_csv(candidate_dir / "recommended_predictions.csv", index=False)
        case_side.to_csv(candidate_dir / "topology_case_side.csv", index=False)
        false_segments.to_csv(candidate_dir / "false_positive_segments.csv", index=False)

        decoder_payload = {
            "status": "validation_only_v1a3b_decoder_candidate",
            "test_set_used": False,
            "name": name,
            "decoder": config.to_dict(),
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
            json.dumps(decoder_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        calibration_payload = {
            "configuration": {
                "variant": "smooth",
                "max_gap": 0,
                "left_threshold": config.left_threshold,
                "right_threshold": config.right_threshold,
            },
            "decoder": config.to_dict(),
            "status": "v1a3b_span_decoder_materialized_predictions",
            "test_set_used": False,
        }
        calibration_path = candidate_dir / "recommended_calibration.json"
        calibration_path.write_text(
            json.dumps(calibration_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        candidates.append(
            {
                "name": name,
                "directory": str(candidate_dir),
                "predictions": str(candidate_dir / "recommended_predictions.csv"),
                "calibration_json": str(calibration_path),
                "expansion_scale": float(args.expansion_scale),
                "decoder": config.to_dict(),
                "presence_metrics": summary,
            }
        )

    manifest_payload = {
        "status": "validation_only_v1a3b_shortlist",
        "test_set_used": False,
        "source_predictions": str(prediction_path),
        "selection": (
            "B0 is the locked V1a.3-A baseline. For B1/B2, retain the lowest-FPR validation "
            "configurations satisfying overall recall >= 0.95 and per-side recall >= 0.90."
        ),
        "candidates": candidates,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_payload = {
        "status": "validation_only_v1a3b_span_sweep",
        "test_set_used": False,
        "num_swept_configs": int(len(sweep)),
        "num_feasible_configs": int(sweep["constraints_met"].sum()),
        "baseline": baseline_row,
        "shortlist": [item["name"] for item in candidates],
    }
    (output_dir / "sweep_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
