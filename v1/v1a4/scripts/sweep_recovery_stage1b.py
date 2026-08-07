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
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a3b"))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a1.calibration import add_threshold_independent_smoothing
from v1a3b.span_decoder import summarize_presence_topology
from v1a4.recovery_decoder import RecoveryDecoderConfig, apply_recovery_decoder
from v1a4.stage1b_decoder import (
    FIXED_LOCAL_HYSTERESIS,
    Stage1bDecoderConfig,
    apply_stage1b_decoder,
)


def parse_float_grid(text: str) -> list[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("Empty float grid")
    return values


def parse_int_grid(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("Empty integer grid")
    return values


def validate_validation_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    frame_keys = set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))
    val_keys = set(zip(val["case_id"].astype(str), val["slice_idx"].astype(int)))
    if frame_keys != val_keys:
        raise RuntimeError("Prediction rows do not exactly match validation split")
    if int(val["case_id"].nunique()) != 20 or len(val) != 3644:
        raise RuntimeError(
            f"Unexpected validation split: cases={val['case_id'].nunique()}, rows={len(val)}"
        )


def metrics(summary: dict) -> dict:
    return {
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


def constraints_met(row: dict, args: argparse.Namespace) -> bool:
    return bool(
        row["positive_recall"] >= args.min_overall_recall
        and row["left_positive_recall"] >= args.min_left_recall
        and row["right_positive_recall"] >= args.min_right_recall
        and row["false_positive_rate"] <= args.max_fpr
    )


def config_key(row: pd.Series) -> tuple:
    return (
        str(row["mode"]),
        int(row["satellite_max_distance"]),
        float(row["satellite_evidence_ratio"]),
        int(row["satellite_min_length"]),
    )


def choose_diverse_rows(feasible: pd.DataFrame, limit: int) -> list[pd.Series]:
    if feasible.empty or limit <= 0:
        return []
    feasible = feasible.copy()
    feasible["boundary_mae"] = feasible[
        ["onset_abs_error_mean", "offset_abs_error_mean"]
    ].mean(axis=1)
    orderings = [
        (["false_positive_rate", "positive_recall", "boundary_mae"], [True, False, True]),
        (["positive_recall", "false_positive_rate", "boundary_mae"], [False, True, True]),
        (["boundary_mae", "false_positive_rate", "positive_recall"], [True, True, False]),
    ]
    picks: list[pd.Series] = []
    seen: set[tuple] = set()
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
    return picks


def stage1b_config_from_row(row: pd.Series) -> Stage1bDecoderConfig:
    return Stage1bDecoderConfig(
        mode=str(row["mode"]),
        satellite_max_distance=int(row["satellite_max_distance"]),
        satellite_evidence_ratio=float(row["satellite_evidence_ratio"]),
        satellite_min_length=int(row["satellite_min_length"]),
        max_satellites=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V1a.4 Stage-1b validation sweep: strongly supported remote satellite "
            "recovery, alone or combined with the fixed Stage-1 local hysteresis rule."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--satellite-max-distances",
        default="20,24,32,48,64,96",
    )
    parser.add_argument(
        "--satellite-evidence-ratios",
        default="0.05,0.1,0.2,0.35,0.5,0.75",
    )
    parser.add_argument("--satellite-min-lengths", default="1,2,3,5,8")
    parser.add_argument("--min-overall-recall", type=float, default=0.96)
    parser.add_argument("--min-left-recall", type=float, default=0.94)
    parser.add_argument("--min-right-recall", type=float, default=0.97)
    parser.add_argument("--max-fpr", type=float, default=0.15)
    parser.add_argument("--shortlist-per-mode", type=int, default=3)
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prediction_path, dtype={"case_id": str})
    validate_validation_rows(frame, manifest_path)
    frame = add_threshold_independent_smoothing(frame)

    reference_configs = {
        "v1a3b_frozen_b1": RecoveryDecoderConfig(mode="frozen_b1"),
        "stage1_fixed_local_hysteresis": RecoveryDecoderConfig(
            mode="hysteresis",
            left_recovery_threshold=float(FIXED_LOCAL_HYSTERESIS["left_recovery_threshold"]),
            right_recovery_threshold=float(FIXED_LOCAL_HYSTERESIS["right_recovery_threshold"]),
            max_extension=int(FIXED_LOCAL_HYSTERESIS["max_extension"]),
            bridge_gap=int(FIXED_LOCAL_HYSTERESIS["bridge_gap"]),
        ),
    }

    reference_rows = []
    for name, config in reference_configs.items():
        decoded = apply_recovery_decoder(frame, config)
        summary, _, _ = summarize_presence_topology(decoded)
        row = {
            "name": name,
            "mode": config.mode,
            **metrics(summary),
            "constraints_met": constraints_met(metrics(summary), args),
        }
        reference_rows.append(row)
    pd.DataFrame(reference_rows).to_csv(output_dir / "reference_metrics.csv", index=False)

    jobs: list[Stage1bDecoderConfig] = []
    for mode, distance, ratio, min_length in itertools.product(
        ("remote_satellite", "hybrid"),
        parse_int_grid(args.satellite_max_distances),
        parse_float_grid(args.satellite_evidence_ratios),
        parse_int_grid(args.satellite_min_lengths),
    ):
        jobs.append(
            Stage1bDecoderConfig(
                mode=mode,
                satellite_max_distance=distance,
                satellite_evidence_ratio=ratio,
                satellite_min_length=min_length,
                max_satellites=1,
            )
        )

    rows = []
    for index, config in enumerate(jobs, start=1):
        decoded = apply_stage1b_decoder(frame, config)
        summary, _, _ = summarize_presence_topology(decoded)
        row = {**config.to_dict(), **metrics(summary)}
        row["constraints_met"] = constraints_met(row, args)
        rows.append(row)
        if index % 100 == 0 or index == len(jobs):
            print(f"Swept {index}/{len(jobs)} Stage-1b configurations")

    sweep = pd.DataFrame(rows).sort_values(
        ["constraints_met", "false_positive_rate", "positive_recall"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    sweep.to_csv(output_dir / "stage1b_sweep.csv", index=False)

    selected: list[tuple[str, str, object]] = []
    for name, config in reference_configs.items():
        selected.append((name, "reference", config))

    for mode, prefix in (
        ("remote_satellite", "r2b_remote_satellite"),
        ("hybrid", "r3_hybrid"),
    ):
        feasible = sweep[(sweep["mode"] == mode) & sweep["constraints_met"]].copy()
        for rank, row in enumerate(
            choose_diverse_rows(feasible, args.shortlist_per_mode), start=1
        ):
            selected.append((f"{prefix}_{rank}", "stage1b", stage1b_config_from_row(row)))

    candidate_specs = []
    for name, family, config in selected:
        candidate_dir = output_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if family == "reference":
            decoded = apply_recovery_decoder(frame, config)
            decoder_payload = config.to_dict()
        else:
            decoded = apply_stage1b_decoder(frame, config)
            decoder_payload = config.to_dict()
        summary, case_side, false_segments = summarize_presence_topology(decoded)
        decoded.to_csv(candidate_dir / "recommended_predictions.csv", index=False)
        case_side.to_csv(candidate_dir / "topology_case_side.csv", index=False)
        false_segments.to_csv(candidate_dir / "false_positive_segments.csv", index=False)

        config_payload = {
            "status": "validation_only_v1a4_stage1b_candidate",
            "test_set_used": False,
            "name": name,
            "family": family,
            "decoder": decoder_payload,
            "fixed_local_hysteresis": FIXED_LOCAL_HYSTERESIS if name.startswith("r3_hybrid") else None,
            "presence_metrics": summary,
            "constraints": {
                "minimum_overall_recall": args.min_overall_recall,
                "minimum_left_recall": args.min_left_recall,
                "minimum_right_recall": args.min_right_recall,
                "maximum_fpr": args.max_fpr,
            },
            "constraints_met": constraints_met(metrics(summary), args),
        }
        (candidate_dir / "decoder_config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        calibration_payload = {
            "configuration": {
                "variant": "smooth",
                "max_gap": 0,
                "left_threshold": 0.0035,
                "right_threshold": 0.04,
            },
            "decoder": decoder_payload,
            "fixed_local_hysteresis": FIXED_LOCAL_HYSTERESIS if name.startswith("r3_hybrid") else None,
            "status": "v1a4_stage1b_materialized_presence_predictions",
            "test_set_used": False,
        }
        calibration_path = candidate_dir / "recommended_calibration.json"
        calibration_path.write_text(
            json.dumps(calibration_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        candidate_specs.append(
            {
                "name": name,
                "family": family,
                "directory": str(candidate_dir),
                "predictions": str(candidate_dir / "recommended_predictions.csv"),
                "calibration_json": str(calibration_path),
                "expansion_scale": float(args.expansion_scale),
                "decoder": decoder_payload,
                "presence_metrics": summary,
            }
        )

    manifest_payload = {
        "status": "validation_only_v1a4_stage1b_shortlist",
        "test_set_used": False,
        "source_predictions": str(prediction_path),
        "stage1_fixed_local_hysteresis": FIXED_LOCAL_HYSTERESIS,
        "selection": (
            "Keep frozen B1 and the fixed Stage-1 local hysteresis point as references. "
            "For remote-satellite and hybrid families, materialize diverse configurations "
            "meeting the unchanged V1a.4 recall/FPR gate."
        ),
        "candidates": candidate_specs,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mode_counts = {
        mode: int(((sweep["mode"] == mode) & sweep["constraints_met"]).sum())
        for mode in ("remote_satellite", "hybrid")
    }
    best_rows = {}
    for mode in ("remote_satellite", "hybrid"):
        subset = sweep[sweep["mode"] == mode].sort_values(
            ["positive_recall", "false_positive_rate"], ascending=[False, True]
        )
        best_rows[mode] = subset.iloc[0].to_dict() if len(subset) else None

    summary_payload = {
        "status": "validation_only_v1a4_stage1b_sweep",
        "test_set_used": False,
        "num_swept_configs": int(len(sweep)),
        "num_feasible_configs": int(sweep["constraints_met"].sum()),
        "feasible_by_mode": mode_counts,
        "fixed_local_hysteresis": FIXED_LOCAL_HYSTERESIS,
        "best_by_recall": best_rows,
        "shortlist": [item["name"] for item in candidate_specs],
    }
    (output_dir / "stage1b_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
