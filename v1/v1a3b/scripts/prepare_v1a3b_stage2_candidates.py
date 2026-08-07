from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a3b.span_decoder import SIDES, SpanDecoderConfig, apply_span_decoder, summarize_presence_topology
from v1a1.calibration import add_threshold_independent_smoothing


def validate_validation_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    frame_keys = set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))
    val_keys = set(zip(val["case_id"].astype(str), val["slice_idx"].astype(int)))
    if frame_keys != val_keys:
        raise RuntimeError("Prediction rows do not exactly match validation split; refusing Stage 2")
    if int(val["case_id"].astype(str).nunique()) != 20 or len(val) != 3644:
        raise RuntimeError(f"Unexpected validation split: cases={val['case_id'].nunique()}, rows={len(val)}")


def pad_single_span_predictions(frame: pd.DataFrame, boundary_pad: int) -> pd.DataFrame:
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
    return pad_single_span_predictions(apply_span_decoder(frame, config), boundary_pad)


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


def choose_low_fpr(sweep: pd.DataFrame, mode: str) -> pd.Series:
    feasible = sweep[(sweep["mode"] == mode) & (sweep["constraints_met"].astype(bool))].copy()
    if feasible.empty:
        raise RuntimeError(f"No feasible {mode} configuration")
    return feasible.sort_values(
        ["false_positive_rate", "positive_recall", "onset_abs_error_mean", "offset_abs_error_mean"],
        ascending=[True, False, True, True],
    ).iloc[0]


def choose_recall_matched(sweep: pd.DataFrame, mode: str, baseline_recall: float) -> pd.Series:
    feasible = sweep[
        (sweep["mode"] == mode)
        & (sweep["constraints_met"].astype(bool))
        & (sweep["positive_recall"] >= float(baseline_recall) - 1e-12)
    ].copy()
    if feasible.empty:
        raise RuntimeError(f"No {mode} configuration matches baseline recall {baseline_recall:.6f}")
    return feasible.sort_values(
        ["false_positive_rate", "positive_recall", "onset_abs_error_mean", "offset_abs_error_mean"],
        ascending=[True, False, True, True],
    ).iloc[0]


def materialize(
    frame: pd.DataFrame,
    output_dir: Path,
    name: str,
    config: SpanDecoderConfig,
    boundary_pad: int,
    expansion_scale: float,
) -> dict:
    candidate_dir = output_dir / "candidates" / name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    decoded = decode_with_padding(frame, config, boundary_pad)
    summary, case_side, false_segments = summarize_presence_topology(decoded)

    predictions_path = candidate_dir / "recommended_predictions.csv"
    calibration_path = candidate_dir / "recommended_calibration.json"
    decoded.to_csv(predictions_path, index=False)
    case_side.to_csv(candidate_dir / "topology_case_side.csv", index=False)
    false_segments.to_csv(candidate_dir / "false_positive_segments.csv", index=False)

    decoder_dict = {**config.to_dict(), "boundary_pad": int(boundary_pad)}
    (candidate_dir / "decoder_config.json").write_text(
        json.dumps(
            {
                "status": "validation_only_v1a3b_stage2_candidate",
                "test_set_used": False,
                "name": name,
                "decoder": decoder_dict,
                "presence_metrics": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calibration_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "variant": "smooth",
                    "max_gap": 0,
                    "left_threshold": config.left_threshold,
                    "right_threshold": config.right_threshold,
                },
                "decoder": decoder_dict,
                "status": "v1a3b_span_decoder_materialized_predictions",
                "test_set_used": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "name": name,
        "directory": str(candidate_dir),
        "predictions": str(predictions_path),
        "calibration_json": str(calibration_path),
        "expansion_scale": float(expansion_scale),
        "decoder": decoder_dict,
        "presence_metrics": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a compact V1a.3-B Stage-2 MedSAM candidate set")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sweep-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, dtype={"case_id": str})
    validate_validation_rows(frame, Path(args.manifest))
    frame = add_threshold_independent_smoothing(frame)
    sweep = pd.read_csv(args.sweep_csv)

    baseline_config = SpanDecoderConfig(
        mode="baseline",
        left_threshold=0.005,
        right_threshold=0.005,
        baseline_max_gap=2,
    )
    baseline_decoded = decode_with_padding(frame, baseline_config, 0)
    baseline_summary, _, _ = summarize_presence_topology(baseline_decoded)
    baseline_recall = float(baseline_summary["positive_recall"])

    selected: list[tuple[str, SpanDecoderConfig, int]] = [
        ("b0_locked_baseline", baseline_config, 0),
    ]
    for name, mode in (
        ("b1_low_fpr", "single_span"),
        ("b2_low_fpr", "optimal_span"),
    ):
        config, pad = config_from_row(choose_low_fpr(sweep, mode))
        selected.append((name, config, pad))
    for name, mode in (
        ("b1_recall_matched", "single_span"),
        ("b2_recall_matched", "optimal_span"),
    ):
        config, pad = config_from_row(choose_recall_matched(sweep, mode, baseline_recall))
        selected.append((name, config, pad))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        materialize(frame, output_dir, name, config, pad, args.expansion_scale)
        for name, config, pad in selected
    ]

    payload = {
        "status": "validation_only_v1a3b_stage2_medsam_candidates",
        "test_set_used": False,
        "selection_rule": (
            "B0 locked baseline plus, for each decoder family, (1) minimum-FPR feasible point and "
            "(2) minimum-FPR point matching or exceeding B0 side-level recall. "
            "No MedSAM result is used to construct this candidate set."
        ),
        "baseline_recall": baseline_recall,
        "candidates": candidates,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for item in candidates:
        m = item["presence_metrics"]
        d = item["decoder"]
        rows.append(
            {
                "name": item["name"],
                "mode": d["mode"],
                "left_threshold": d["left_threshold"],
                "right_threshold": d["right_threshold"],
                "baseline_max_gap": d["baseline_max_gap"],
                "max_internal_gap": d["max_internal_gap"],
                "boundary_pad": d["boundary_pad"],
                "positive_recall": m["positive_recall"],
                "left_positive_recall": m["left_positive_recall"],
                "right_positive_recall": m["right_positive_recall"],
                "false_positive_rate": m["false_positive_rate"],
                "specificity": m["specificity"],
                "onset_abs_error_mean": m["onset_abs_error_mean"],
                "offset_abs_error_mean": m["offset_abs_error_mean"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "stage2_presence_candidates.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
