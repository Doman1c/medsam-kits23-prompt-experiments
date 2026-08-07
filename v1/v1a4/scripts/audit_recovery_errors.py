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
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a3b"))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a1.calibration import add_threshold_independent_smoothing
from v1a3b.span_decoder import SIDES, summarize_presence_topology
from v1a4.recovery_decoder import RecoveryDecoderConfig, apply_recovery_decoder


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit which positive kidney slices are removed by frozen V1a.3-B B1"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prediction_path, dtype={"case_id": str})
    validate_validation_rows(frame, manifest_path)
    frame = add_threshold_independent_smoothing(frame)
    decoded = apply_recovery_decoder(
        frame,
        RecoveryDecoderConfig(mode="frozen_b1"),
    )
    presence_summary, _, _ = summarize_presence_topology(decoded)

    miss_rows: list[dict] = []
    case_side_rows: list[dict] = []
    ordered = decoded.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)

    for case_id, case in ordered.groupby("case_id", sort=False):
        case = case.sort_values("slice_idx")
        z = case["slice_idx"].to_numpy(dtype=int)
        for side in SIDES:
            gt = case[f"{side}_gt_presence"].to_numpy(dtype=int).astype(bool)
            pred = case[f"{side}_smooth_presence"].to_numpy(dtype=int).astype(bool)
            fn = np.logical_and(gt, ~pred)
            pred_pos = np.flatnonzero(pred)
            fn_pos = np.flatnonzero(fn)

            before = after = inside = no_seed = 0
            distances: list[int] = []
            if len(pred_pos):
                pred_start = int(pred_pos[0])
                pred_end = int(pred_pos[-1])
            else:
                pred_start = pred_end = -1

            for local_index in fn_pos:
                if not len(pred_pos):
                    relation = "no_seed"
                    distance = None
                    no_seed += 1
                else:
                    distance = int(np.abs(pred_pos - local_index).min())
                    distances.append(distance)
                    if local_index < pred_start:
                        relation = "before_seed"
                        before += 1
                    elif local_index > pred_end:
                        relation = "after_seed"
                        after += 1
                    else:
                        relation = "inside_seed_extent"
                        inside += 1
                miss_rows.append(
                    {
                        "case_id": str(case_id),
                        "side": side,
                        "slice_idx": int(z[local_index]),
                        "relation": relation,
                        "distance_to_seed": distance,
                        "smooth_presence_probability": float(
                            case.iloc[local_index][f"{side}_smooth_presence_probability"]
                        ),
                    }
                )

            case_side_rows.append(
                {
                    "case_id": str(case_id),
                    "side": side,
                    "gt_positive_slices": int(gt.sum()),
                    "seed_positive_slices": int(pred.sum()),
                    "false_negative_slices": int(fn.sum()),
                    "fn_before_seed": int(before),
                    "fn_after_seed": int(after),
                    "fn_inside_seed_extent": int(inside),
                    "fn_no_seed": int(no_seed),
                    "max_fn_distance": int(max(distances)) if distances else None,
                    "mean_fn_distance": float(np.mean(distances)) if distances else None,
                }
            )

    miss = pd.DataFrame(miss_rows)
    case_side = pd.DataFrame(case_side_rows)

    distance_counts = {}
    finite_distance = (
        pd.to_numeric(miss["distance_to_seed"], errors="coerce")
        if len(miss)
        else pd.Series(dtype=float)
    )
    total_misses = int(len(miss))
    for distance in (1, 2, 4, 6, 8, 12, 16):
        count = int((finite_distance <= distance).sum())
        distance_counts[str(distance)] = {
            "count": count,
            "fraction_of_all_false_negative_slices": (
                float(count / total_misses) if total_misses else 0.0
            ),
        }

    relation_counts = (
        miss["relation"].value_counts().to_dict() if len(miss) else {}
    )
    side_counts = (
        miss["side"].value_counts().to_dict() if len(miss) else {}
    )

    payload = {
        "status": "validation_only_v1a4_recovery_audit",
        "test_set_used": False,
        "source_predictions": str(prediction_path),
        "frozen_b1_presence": presence_summary,
        "false_negative_slices": total_misses,
        "false_negative_relation_counts": {
            str(key): int(value) for key, value in relation_counts.items()
        },
        "false_negative_side_counts": {
            str(key): int(value) for key, value in side_counts.items()
        },
        "distance_from_frozen_seed": distance_counts,
        "case_sides_with_any_false_negative": int(
            (case_side["false_negative_slices"] > 0).sum()
        ),
        "case_sides_with_no_seed_false_negative": int(
            (case_side["fn_no_seed"] > 0).sum()
        ),
    }

    miss.to_csv(output_dir / "recovery_miss_details.csv", index=False)
    case_side.to_csv(output_dir / "recovery_case_side.csv", index=False)
    (output_dir / "recovery_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
