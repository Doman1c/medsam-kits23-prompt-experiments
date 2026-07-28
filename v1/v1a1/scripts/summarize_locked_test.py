from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
# When copied to v1/v1a1/scripts/, parents[1] is v1/v1a1.
# When executed elsewhere, run it from the v1/v1a1 directory or set PYTHONPATH.
if (ROOT / "v1a1").exists():
    sys.path.insert(0, str(ROOT))
else:
    cwd = Path.cwd()
    if (cwd / "v1a1").exists():
        sys.path.insert(0, str(cwd))

from v1a1.data import load_npz  # noqa: E402
from v1a1.inference import expand_box  # noqa: E402
from v1a1.metrics import evaluate_prediction_frame, write_evaluation  # noqa: E402

LOCK = {
    "epoch": 6,
    "weights": "ema",
    "variant": "smooth",
    "max_gap": 0,
    "left_threshold": 0.20,
    "right_threshold": 0.30,
    "expansion_scale": 1.20,
    "target_labels": [1],
}
AREA_ORDER = ["00_10", "10_50", "50_90", "90_100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the frozen V1a.1 test evaluation without tuning."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True, help="Calibrated test prediction CSV")
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--medsam-details", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-test-cases", type=int, default=27)
    parser.add_argument("--expected-test-rows", type=int, default=4277)
    parser.add_argument("--expected-positive-slices", type=int, default=1473)
    parser.add_argument("--figure-count", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def assert_close(name: str, actual: float, expected: float, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0.0):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def load_and_validate_lock(args: argparse.Namespace) -> tuple[dict, dict]:
    payload = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    cfg = payload.get("configuration", payload)
    if str(cfg["variant"]) != LOCK["variant"]:
        raise AssertionError(f"variant: expected {LOCK['variant']}, got {cfg['variant']}")
    if int(cfg.get("max_gap", 0)) != LOCK["max_gap"]:
        raise AssertionError(f"max_gap: expected {LOCK['max_gap']}, got {cfg.get('max_gap')}")
    assert_close("left_threshold", cfg["left_threshold"], LOCK["left_threshold"])
    assert_close("right_threshold", cfg["right_threshold"], LOCK["right_threshold"])

    if "candidate" in payload and str(payload["candidate"]).lower() != "epoch6":
        raise AssertionError(f"candidate: expected epoch6, got {payload['candidate']}")
    if "expansion_scale" in payload:
        assert_close("expansion_scale", payload["expansion_scale"], LOCK["expansion_scale"])

    checkpoint = read_checkpoint(args.checkpoint)
    if int(checkpoint.get("epoch", -1)) != LOCK["epoch"]:
        raise AssertionError(
            f"checkpoint epoch: expected {LOCK['epoch']}, got {checkpoint.get('epoch')}"
        )
    if "ema_state" not in checkpoint:
        raise AssertionError("Checkpoint has no ema_state, but the locked evaluation requires EMA weights")

    config = checkpoint.get("config", {})
    if str(config.get("input_mode", "2.5d")) != "2.5d":
        raise AssertionError(f"input_mode is not 2.5d: {config.get('input_mode')}")
    return payload, checkpoint


def normalize_keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))


def validate_test_rows(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    expected_cases: int,
    expected_rows: int,
    expected_positive: int,
) -> dict[str, int]:
    required_pred = {
        "case_id",
        "slice_idx",
        "npz_path",
        "left_gt_presence",
        "right_gt_presence",
        "left_smooth_presence",
        "right_smooth_presence",
        "combined_area_bin",
        "combined_relative_area",
    }
    missing = sorted(required_pred - set(predictions.columns))
    if missing:
        raise AssertionError(f"Calibrated predictions are missing columns: {missing}")

    if predictions.duplicated(["case_id", "slice_idx"]).any():
        raise AssertionError("Duplicate (case_id, slice_idx) rows in predictions")

    if "split" not in manifest.columns:
        raise AssertionError("Manifest has no split column")
    test_manifest = manifest[manifest["split"].astype(str) == "test"].copy()
    if test_manifest.empty:
        raise AssertionError("Manifest contains no test rows")

    counts = {
        "manifest_test_rows": int(len(test_manifest)),
        "prediction_rows": int(len(predictions)),
        "test_cases": int(predictions["case_id"].astype(str).nunique()),
        "positive_slices": int(
            ((predictions["left_gt_presence"] == 1) | (predictions["right_gt_presence"] == 1)).sum()
        ),
    }
    expected = {
        "manifest_test_rows": expected_rows,
        "prediction_rows": expected_rows,
        "test_cases": expected_cases,
        "positive_slices": expected_positive,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise AssertionError(f"{key}: expected {value}, got {counts[key]}")

    manifest_keys = normalize_keys(test_manifest)
    prediction_keys = normalize_keys(predictions)
    if manifest_keys != prediction_keys:
        missing_pred = sorted(manifest_keys - prediction_keys)[:10]
        extra_pred = sorted(prediction_keys - manifest_keys)[:10]
        raise AssertionError(
            "Prediction/test-manifest row mismatch. "
            f"Missing predictions (first 10): {missing_pred}; extra predictions (first 10): {extra_pred}"
        )
    return counts


def mode_frame(details: pd.DataFrame, mode: str, prefix: str) -> pd.DataFrame:
    subset = details[details["mode"] == mode].copy()
    if subset.empty:
        raise AssertionError(f"MedSAM details have no mode={mode}")
    keep = [
        "case_id",
        "slice_idx",
        "npz_path",
        "combined_area_bin",
        "combined_relative_area",
        "num_prompts",
        "dice",
        "iou",
    ]
    subset = subset[keep]
    return subset.rename(
        columns={
            "num_prompts": f"{prefix}_num_prompts",
            "dice": f"{prefix}_dice",
            "iou": f"{prefix}_iou",
        }
    )


def build_medsam_tables(details: pd.DataFrame, expected_positive: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    auto = mode_frame(details, "dual_smooth", "automatic")
    oracle_scaled = mode_frame(details, "oracle_scaled", "oracle_scaled")
    oracle_tight = mode_frame(details, "oracle_tight", "oracle_tight")

    for name, frame in {
        "dual_smooth": auto,
        "oracle_scaled": oracle_scaled,
        "oracle_tight": oracle_tight,
    }.items():
        if len(frame) != expected_positive:
            raise AssertionError(f"{name}: expected {expected_positive} rows, got {len(frame)}")
        if frame.duplicated(["case_id", "slice_idx"]).any():
            raise AssertionError(f"Duplicate rows in MedSAM mode={name}")

    merge_keys = [
        "case_id",
        "slice_idx",
        "npz_path",
        "combined_area_bin",
        "combined_relative_area",
    ]
    per_slice = auto.merge(oracle_scaled, on=merge_keys, how="inner", validate="one_to_one")
    per_slice = per_slice.merge(oracle_tight, on=merge_keys, how="inner", validate="one_to_one")
    if len(per_slice) != expected_positive:
        raise AssertionError(f"Merged per-slice rows: expected {expected_positive}, got {len(per_slice)}")
    per_slice["automatic_to_oracle_gap"] = (
        per_slice["oracle_scaled_dice"] - per_slice["automatic_dice"]
    )
    per_slice = per_slice.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)

    per_case = (
        per_slice.groupby("case_id", as_index=False)
        .agg(
            positive_slices=("slice_idx", "size"),
            automatic_dice_mean=("automatic_dice", "mean"),
            automatic_dice_median=("automatic_dice", "median"),
            automatic_iou_mean=("automatic_iou", "mean"),
            oracle_scaled_dice_mean=("oracle_scaled_dice", "mean"),
            oracle_scaled_dice_median=("oracle_scaled_dice", "median"),
            oracle_tight_dice_mean=("oracle_tight_dice", "mean"),
            oracle_gap_mean=("automatic_to_oracle_gap", "mean"),
            oracle_gap_median=("automatic_to_oracle_gap", "median"),
        )
        .sort_values("automatic_dice_mean")
        .reset_index(drop=True)
    )

    area_summary = (
        per_slice.groupby("combined_area_bin", as_index=False)
        .agg(
            n_slices=("slice_idx", "size"),
            automatic_dice_mean=("automatic_dice", "mean"),
            automatic_dice_median=("automatic_dice", "median"),
            automatic_iou_mean=("automatic_iou", "mean"),
            oracle_scaled_dice_mean=("oracle_scaled_dice", "mean"),
            oracle_scaled_dice_median=("oracle_scaled_dice", "median"),
            oracle_gap_mean=("automatic_to_oracle_gap", "mean"),
            oracle_gap_median=("automatic_to_oracle_gap", "median"),
        )
    )
    area_summary["_order"] = area_summary["combined_area_bin"].map(
        {name: i for i, name in enumerate(AREA_ORDER)}
    )
    area_summary = area_summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return per_slice, per_case, area_summary


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def area_records(area_summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _, row in area_summary.iterrows():
        key = str(row["combined_area_bin"])
        result[key] = {column: scalar(row[column]) for column in area_summary.columns if column != "combined_area_bin"}
    return result


def plot_prompt_overlay(
    prediction_row: pd.Series,
    score_row: pd.Series,
    output_path: Path,
    category: str,
) -> None:
    image, label = load_npz(prediction_row["npz_path"])
    gt = label == 1
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray")
    if gt.any():
        ax.contour(gt.astype(float), levels=[0.5], colors=["lime"], linewidths=1.0)

    for side, color in (("left", "cyan"), ("right", "magenta")):
        if int(prediction_row[f"{side}_smooth_presence"]) != 1:
            continue
        box = tuple(float(prediction_row[f"{side}_smooth_{key}"]) for key in ("x1", "y1", "x2", "y2"))
        box = expand_box(box, LOCK["expansion_scale"], int(prediction_row["image_width"]), int(prediction_row["image_height"]))
        x1, y1, x2, y2 = box
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5)
        )
        ax.text(x1, max(0, y1 - 3), side, color=color, fontsize=8, backgroundcolor="black")

    ax.set_title(
        f"{category} | {score_row['case_id']} z={int(score_row['slice_idx'])}\n"
        f"auto Dice={score_row['automatic_dice']:.4f}, "
        f"oracle={score_row['oracle_scaled_dice']:.4f}, "
        f"gap={score_row['automatic_to_oracle_gap']:.4f}, "
        f"area={score_row['combined_area_bin']}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_representative_figures(
    predictions: pd.DataFrame,
    per_slice: pd.DataFrame,
    output_dir: Path,
    count: int,
) -> None:
    if count <= 0:
        return
    figure_dir = output_dir / "representative_prompt_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    selections: list[tuple[str, pd.DataFrame]] = [
        ("success", per_slice.nlargest(count, "automatic_dice")),
        ("failure", per_slice.nsmallest(count, "automatic_dice")),
        ("largest_gap", per_slice.nlargest(count, "automatic_to_oracle_gap")),
    ]
    small = per_slice[per_slice["combined_area_bin"] == "00_10"]
    if len(small):
        selections.append(("small_area_failure", small.nsmallest(count, "automatic_dice")))

    prediction_index = predictions.set_index([predictions["case_id"].astype(str), predictions["slice_idx"].astype(int)])
    used: set[tuple[str, int, str]] = set()
    for category, rows in selections:
        rank = 0
        for _, score_row in rows.iterrows():
            case_id = str(score_row["case_id"])
            slice_idx = int(score_row["slice_idx"])
            key = (case_id, slice_idx, category)
            if key in used:
                continue
            used.add(key)
            rank += 1
            prediction_row = prediction_index.loc[(case_id, slice_idx)]
            safe_case = case_id.replace("/", "_")
            output_path = figure_dir / f"{category}_{rank:02d}_{safe_case}_z{slice_idx:04d}.png"
            plot_prompt_overlay(prediction_row, score_row, output_path, category)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_payload, checkpoint = load_and_validate_lock(args)
    manifest = pd.read_csv(args.manifest)
    predictions = pd.read_csv(args.predictions)
    counts = validate_test_rows(
        manifest,
        predictions,
        expected_cases=args.expected_test_cases,
        expected_rows=args.expected_test_rows,
        expected_positive=args.expected_positive_slices,
    )

    prompt_summary, prompt_details = evaluate_prediction_frame(
        predictions,
        variant="smooth",
        expansion_scale=LOCK["expansion_scale"],
        target_labels=tuple(LOCK["target_labels"]),
    )
    write_evaluation(prompt_summary, prompt_details, output_dir, "prompt_test")

    medsam_details = pd.read_csv(args.medsam_details)
    per_slice, per_case, area_summary = build_medsam_tables(
        medsam_details, expected_positive=args.expected_positive_slices
    )
    per_slice.to_csv(output_dir / "test_per_slice.csv", index=False)
    per_case.to_csv(output_dir / "test_per_case.csv", index=False)
    area_summary.to_csv(output_dir / "test_area_bin_summary.csv", index=False)
    per_slice.nsmallest(50, "automatic_dice").to_csv(
        output_dir / "test_low_dice_ranked.csv", index=False
    )
    per_slice.nlargest(50, "automatic_to_oracle_gap").to_csv(
        output_dir / "test_largest_oracle_gap_ranked.csv", index=False
    )

    summary = {
        "status": "frozen_test_evaluation",
        "locked_configuration": LOCK,
        "git_commit": current_git_commit(),
        "data_counts": counts,
        "checkpoint_metadata": {
            "epoch": int(checkpoint["epoch"]),
            "input_mode": checkpoint.get("config", {}).get("input_mode"),
            "input_size": checkpoint.get("config", {}).get("input_size"),
            "heatmap_size": checkpoint.get("config", {}).get("heatmap_size"),
            "base_channels": checkpoint.get("config", {}).get("base_channels"),
        },
        "artifact_sha256": {
            "manifest": sha256_file(args.manifest),
            "checkpoint": sha256_file(args.checkpoint),
            "calibration_json": sha256_file(args.calibration_json),
            "calibrated_predictions": sha256_file(args.predictions),
            "medsam_details": sha256_file(args.medsam_details),
        },
        "calibration_source": calibration_payload,
        "prompt_metrics": prompt_summary,
        "medsam_metrics": {
            "automatic_mode": "dual_smooth",
            "oracle_mode": "oracle_scaled",
            "positive_slices_only": True,
            "slice_level": {
                "mean_dice": float(per_slice["automatic_dice"].mean()),
                "median_dice": float(per_slice["automatic_dice"].median()),
                "mean_iou": float(per_slice["automatic_iou"].mean()),
                "oracle_mean_dice": float(per_slice["oracle_scaled_dice"].mean()),
                "oracle_median_dice": float(per_slice["oracle_scaled_dice"].median()),
                "oracle_gap_mean": float(per_slice["automatic_to_oracle_gap"].mean()),
                "oracle_gap_median": float(per_slice["automatic_to_oracle_gap"].median()),
            },
            "case_level": {
                "n_cases": int(len(per_case)),
                "mean_dice": float(per_case["automatic_dice_mean"].mean()),
                "median_dice": float(per_case["automatic_dice_mean"].median()),
                "mean_of_case_medians": float(per_case["automatic_dice_median"].mean()),
                "oracle_mean_dice": float(per_case["oracle_scaled_dice_mean"].mean()),
                "oracle_gap_mean": float(per_case["oracle_gap_mean"].mean()),
                "oracle_gap_median": float(per_case["oracle_gap_mean"].median()),
            },
            "by_combined_area_bin": area_records(area_summary),
        },
        "reporting_note": (
            "Test outputs are descriptive only. No checkpoint, threshold, smoothing, gap filling, "
            "or expansion choice may be changed in response to these results."
        ),
    }
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_representative_figures(
        predictions, per_slice, output_dir, count=args.figure_count
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved frozen test report to: {output_dir}")


if __name__ == "__main__":
    main()
