from __future__ import annotations

import math


def _finite(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


def presence_checkpoint_score(
    metrics: dict,
    *,
    min_recall: float = 0.95,
    min_left_recall: float = 0.90,
    min_right_recall: float = 0.90,
) -> tuple[float, dict]:
    recall = _finite(metrics.get("val_positive_recall", 0.0))
    left = _finite(metrics.get("val_left_positive_recall", 0.0))
    right = _finite(metrics.get("val_right_positive_recall", 0.0))
    fpr = _finite(metrics.get("val_false_positive_rate", 1.0), 1.0)
    specificity = 1.0 - min(max(fpr, 0.0), 1.0)

    feasible = (
        recall >= min_recall
        and left >= min_left_recall
        and right >= min_right_recall
    )
    base = (
        0.45 * recall
        + 0.15 * left
        + 0.15 * right
        + 0.25 * specificity
    )
    penalty = (
        max(min_recall - recall, 0.0)
        + 0.5 * max(min_left_recall - left, 0.0)
        + 0.5 * max(min_right_recall - right, 0.0)
    )
    score = base if feasible else base - 2.0 * penalty
    return float(score), {
        "feasible": bool(feasible),
        "recall": recall,
        "left_recall": left,
        "right_recall": right,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "penalty": float(penalty),
    }


def geometry_checkpoint_score(metrics: dict) -> float:
    return _finite(metrics.get("val_continuous_score", -1.0), -1.0)


def joint_checkpoint_score(
    metrics: dict,
    *,
    min_recall: float = 0.95,
    min_left_recall: float = 0.90,
    min_right_recall: float = 0.90,
) -> tuple[float, dict]:
    presence_score, details = presence_checkpoint_score(
        metrics,
        min_recall=min_recall,
        min_left_recall=min_left_recall,
        min_right_recall=min_right_recall,
    )
    prompt_success = _finite(metrics.get("val_prompt_success_rate", 0.0))
    tight_iou = _finite(metrics.get("val_tight_box_iou", 0.0))
    continuous = _finite(metrics.get("val_continuous_score", 0.0))
    score = (
        0.50 * presence_score
        + 0.25 * continuous
        + 0.15 * prompt_success
        + 0.10 * tight_iou
    )
    details = details | {
        "presence_score": presence_score,
        "continuous_score": continuous,
        "prompt_success_rate": prompt_success,
        "tight_box_iou": tight_iou,
    }
    return float(score), details
