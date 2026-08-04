from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .components import match_component_to_box
from .data import load_npz
from .inference import box_iou, expand_box


SIDES = ("left", "right")


def pixels_inside_box(mask: np.ndarray, box: Iterable[float]) -> int:
    x1, y1, x2, y2 = map(float, box)
    height, width = mask.shape
    ix1 = int(np.clip(np.floor(x1), 0, width))
    iy1 = int(np.clip(np.floor(y1), 0, height))
    ix2 = int(np.clip(np.ceil(x2), 0, width))
    iy2 = int(np.clip(np.ceil(y2), 0, height))
    return int(mask[iy1:iy2, ix1:ix2].sum())


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    variant: str = "raw",
    expansion_scale: float = 1.15,
    target_labels: tuple[int, ...] = (1,),
    presence_threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    details: list[dict] = []
    presence_tp = presence_fn = presence_fp = presence_tn = 0
    order_violations = 0
    both_predicted = 0

    for _, row in frame.iterrows():
        _, label = load_npz(row["npz_path"])
        target_mask = np.isin(label, np.asarray(target_labels)).astype(np.uint8)
        predicted_centers: dict[str, float] = {}
        for side in SIDES:
            gt_present = int(row[f"{side}_gt_presence"])
            probability = float(row[f"{side}_{variant}_presence_probability"])
            pred_present = int(row.get(f"{side}_{variant}_presence", probability >= presence_threshold))
            if gt_present and pred_present:
                presence_tp += 1
            elif gt_present and not pred_present:
                presence_fn += 1
            elif not gt_present and pred_present:
                presence_fp += 1
            else:
                presence_tn += 1

            record = {
                "case_id": row["case_id"],
                "slice_idx": int(row["slice_idx"]),
                "side": side,
                "area_bin": row[f"{side}_area_bin"],
                "gt_present": gt_present,
                "pred_present": pred_present,
                "presence_probability": probability,
            }
            if not gt_present:
                record.update(
                    center_error_pixels=np.nan,
                    center_error_fraction=np.nan,
                    tight_box_iou=np.nan,
                    expanded_coverage=np.nan,
                    box_purity=np.nan,
                    prompt_success=np.nan,
                )
                details.append(record)
                continue

            gt_box = [float(row[f"{side}_gt_{key}"]) for key in ("x1", "y1", "x2", "y2")]
            gt_mask = match_component_to_box(target_mask, gt_box, min_area=1)
            gt_area = max(int(gt_mask.sum()), 1)
            if pred_present:
                pred_box = [float(row[f"{side}_{variant}_{key}"]) for key in ("x1", "y1", "x2", "y2")]
                expanded = expand_box(
                    pred_box,
                    expansion_scale,
                    int(row["image_width"]),
                    int(row["image_height"]),
                )
                gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
                pred_cx, pred_cy = (pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0
                gt_width = max(gt_box[2] - gt_box[0], 1.0)
                gt_height = max(gt_box[3] - gt_box[1], 1.0)
                center_error_pixels = float(((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5)
                center_error_fraction = float(
                    max(abs(pred_cx - gt_cx) / gt_width, abs(pred_cy - gt_cy) / gt_height)
                )
                covered = pixels_inside_box(gt_mask, expanded)
                coverage = covered / gt_area
                expanded_area = max((expanded[2] - expanded[0]) * (expanded[3] - expanded[1]), 1.0)
                purity = covered / expanded_area
                iou = box_iou(pred_box, gt_box)
                success = int(center_error_fraction <= 0.10 and coverage >= 0.95)
                predicted_centers[side] = pred_cx
            else:
                center_error_pixels = float("inf")
                center_error_fraction = float("inf")
                coverage = 0.0
                purity = 0.0
                iou = 0.0
                success = 0

            record.update(
                center_error_pixels=center_error_pixels,
                center_error_fraction=center_error_fraction,
                tight_box_iou=iou,
                expanded_coverage=coverage,
                box_purity=purity,
                prompt_success=success,
            )
            details.append(record)

        if "left" in predicted_centers and "right" in predicted_centers:
            both_predicted += 1
            if predicted_centers["left"] >= predicted_centers["right"]:
                order_violations += 1

    detail_frame = pd.DataFrame(details)
    positive = detail_frame[detail_frame["gt_present"] == 1].copy()
    valid_center = positive.replace([np.inf, -np.inf], np.nan)["center_error_fraction"].dropna()
    recall = presence_tp / max(presence_tp + presence_fn, 1)
    fpr = presence_fp / max(presence_fp + presence_tn, 1)
    accuracy = (presence_tp + presence_tn) / max(presence_tp + presence_fn + presence_fp + presence_tn, 1)

    summary = {
        "variant": variant,
        "expansion_scale": float(expansion_scale),
        "presence_accuracy": float(accuracy),
        "positive_recall": float(recall),
        "false_positive_rate": float(fpr),
        "center_error_pixels_median": float(positive["center_error_pixels"].replace(np.inf, np.nan).median()),
        "center_error_fraction_mean": float(valid_center.mean()) if len(valid_center) else float("nan"),
        "center_error_fraction_median": float(valid_center.median()) if len(valid_center) else float("nan"),
        "center_within_05": float((positive["center_error_fraction"] <= 0.05).mean()),
        "center_within_10": float((positive["center_error_fraction"] <= 0.10).mean()),
        "center_over_20": float((positive["center_error_fraction"] > 0.20).mean()),
        "tight_box_iou": float(positive["tight_box_iou"].mean()),
        "expanded_coverage_mean": float(positive["expanded_coverage"].mean()),
        "box_purity_mean": float(positive["box_purity"].mean()),
        "prompt_success_rate": float(positive["prompt_success"].mean()),
        "side_order_violation_rate": float(order_violations / max(both_predicted, 1)),
        "positive_side_samples": int(len(positive)),
    }

    for side in SIDES:
        side_positive = positive[positive["side"] == side]
        summary[f"{side}_positive_recall"] = float(
            ((side_positive["pred_present"] == 1).sum()) / max(len(side_positive), 1)
        )
        summary[f"{side}_prompt_success"] = float(side_positive["prompt_success"].mean())

    for area_bin in ("00_10", "10_50", "50_90", "90_100"):
        subset = positive[positive["area_bin"] == area_bin]
        summary[f"prompt_success_{area_bin}"] = float(subset["prompt_success"].mean()) if len(subset) else float("nan")
        summary[f"coverage_{area_bin}"] = float(subset["expanded_coverage"].mean()) if len(subset) else float("nan")

    center_quality = 1.0 - float(np.clip(positive["center_error_fraction"].replace(np.inf, 1.0), 0.0, 1.0).mean())
    summary["center_quality"] = center_quality
    summary["continuous_score"] = float(
        0.35 * summary["expanded_coverage_mean"]
        + 0.25 * center_quality
        + 0.20 * summary["positive_recall"]
        + 0.10 * summary["tight_box_iou"]
        - 0.10 * summary["false_positive_rate"]
        - 0.10 * summary["side_order_violation_rate"]
    )
    return summary, detail_frame


def write_evaluation(
    summary: dict,
    details: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> None:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    details.to_csv(output_dir / f"{prefix}_details.csv", index=False)
