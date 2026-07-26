from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .inference import rolling_median

SIDES = ("left", "right")
AREA_BINS = ("00_10", "10_50", "50_90", "90_100")


@dataclass(frozen=True)
class CalibrationConfig:
    variant: str
    max_gap: int
    left_threshold: float
    right_threshold: float

    def threshold_for(self, side: str) -> float:
        return self.left_threshold if side == "left" else self.right_threshold

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "max_gap": int(self.max_gap),
            "left_threshold": float(self.left_threshold),
            "right_threshold": float(self.right_threshold),
        }


def parse_float_grid(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("Threshold grid is empty")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Thresholds must be in [0, 1], got {values}")
    return values


def parse_int_grid(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("Integer grid is empty")
    if any(value < 0 for value in values):
        raise ValueError(f"Values must be non-negative, got {values}")
    return values


def _boxes_from_norms(frame: pd.DataFrame, side: str, prefix: str) -> None:
    width_px = frame[f"{side}_{prefix}_width_norm"].to_numpy(dtype=float) * frame["image_width"].to_numpy(dtype=float)
    height_px = frame[f"{side}_{prefix}_height_norm"].to_numpy(dtype=float) * frame["image_height"].to_numpy(dtype=float)
    cx = frame[f"{side}_{prefix}_center_x_norm"].to_numpy(dtype=float) * frame["image_width"].to_numpy(dtype=float)
    cy = frame[f"{side}_{prefix}_center_y_norm"].to_numpy(dtype=float) * frame["image_height"].to_numpy(dtype=float)
    image_width = frame["image_width"].to_numpy(dtype=float)
    image_height = frame["image_height"].to_numpy(dtype=float)
    frame[f"{side}_{prefix}_x1"] = np.clip(cx - width_px / 2.0, 0.0, image_width)
    frame[f"{side}_{prefix}_y1"] = np.clip(cy - height_px / 2.0, 0.0, image_height)
    frame[f"{side}_{prefix}_x2"] = np.clip(cx + width_px / 2.0, 0.0, image_width)
    frame[f"{side}_{prefix}_y2"] = np.clip(cy + height_px / 2.0, 0.0, image_height)


def add_threshold_independent_smoothing(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 3-slice median-smoothed probabilities, centers and sizes.

    Gap filling and presence thresholding are intentionally deferred to calibration.
    """
    frame = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    for side in SIDES:
        frame[f"{side}_smooth_presence_probability"] = 0.0
        for name in ("center_x_norm", "center_y_norm", "width_norm", "height_norm"):
            frame[f"{side}_smooth_{name}"] = 0.0

    for _, indices in frame.groupby("case_id", sort=False).groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        case = frame.loc[index_array].sort_values("slice_idx")
        ordered_indices = case.index.to_numpy(dtype=int)
        for side in SIDES:
            probability = case[f"{side}_raw_presence_probability"].to_numpy(dtype=float)
            centers = case[[f"{side}_raw_center_x_norm", f"{side}_raw_center_y_norm"]].to_numpy(dtype=float)
            sizes = case[[f"{side}_raw_width_norm", f"{side}_raw_height_norm"]].to_numpy(dtype=float)
            probability_smooth = rolling_median(probability[:, None], window=3)[:, 0]
            center_smooth = rolling_median(centers, window=3)
            log_size_smooth = rolling_median(np.log(np.clip(sizes, 1e-5, None)), window=3)
            size_smooth = np.exp(log_size_smooth)
            frame.loc[ordered_indices, f"{side}_smooth_presence_probability"] = probability_smooth
            frame.loc[ordered_indices, f"{side}_smooth_center_x_norm"] = center_smooth[:, 0]
            frame.loc[ordered_indices, f"{side}_smooth_center_y_norm"] = center_smooth[:, 1]
            frame.loc[ordered_indices, f"{side}_smooth_width_norm"] = size_smooth[:, 0]
            frame.loc[ordered_indices, f"{side}_smooth_height_norm"] = size_smooth[:, 1]

    for side in SIDES:
        _boxes_from_norms(frame, side, "smooth")
    return frame


def _postprocess_presence_case(probability: np.ndarray, threshold: float, max_gap: int) -> np.ndarray:
    present = probability >= float(threshold)
    # Remove isolated one-slice positives.
    for i in range(1, len(present) - 1):
        if present[i] and not present[i - 1] and not present[i + 1]:
            present[i] = False
    # Fill bounded short gaps.
    i = 0
    while i < len(present):
        if present[i]:
            i += 1
            continue
        start = i
        while i < len(present) and not present[i]:
            i += 1
        gap = i - start
        if start > 0 and i < len(present) and gap <= max_gap:
            present[start:i] = True
    return present.astype(np.uint8)


def apply_calibration(frame: pd.DataFrame, config: CalibrationConfig) -> pd.DataFrame:
    if config.variant not in {"raw", "smooth"}:
        raise ValueError(f"Unknown variant: {config.variant}")
    output = frame.copy()
    if config.variant == "raw":
        for side in SIDES:
            probability = output[f"{side}_raw_presence_probability"].to_numpy(dtype=float)
            output[f"{side}_raw_presence"] = (probability >= config.threshold_for(side)).astype(np.uint8)
        return output

    required = f"left_smooth_presence_probability"
    if required not in output.columns:
        output = add_threshold_independent_smoothing(output)
    output = output.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
    for side in SIDES:
        output[f"{side}_smooth_presence"] = 0
    for _, indices in output.groupby("case_id", sort=False).groups.items():
        ordered = output.loc[list(indices)].sort_values("slice_idx")
        ordered_indices = ordered.index.to_numpy(dtype=int)
        for side in SIDES:
            probability = ordered[f"{side}_smooth_presence_probability"].to_numpy(dtype=float)
            present = _postprocess_presence_case(
                probability,
                threshold=config.threshold_for(side),
                max_gap=int(config.max_gap),
            )
            output.loc[ordered_indices, f"{side}_smooth_presence"] = present
    for side in SIDES:
        output[f"{side}_smooth_presence"] = output[f"{side}_smooth_presence"].astype(np.uint8)
    return output


def load_config(path: str | Path) -> CalibrationConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("configuration", payload)
    return CalibrationConfig(
        variant=str(config["variant"]),
        max_gap=int(config.get("max_gap", 0)),
        left_threshold=float(config["left_threshold"]),
        right_threshold=float(config["right_threshold"]),
    )


def save_config(path: str | Path, config: CalibrationConfig, metrics: dict, constraints: dict, feasible: bool) -> None:
    payload = {
        "configuration": config.to_dict(),
        "validation_metrics": metrics,
        "constraints": constraints,
        "constraints_met": bool(feasible),
        "selection_rule": (
            "Among configurations meeting recall constraints, maximize prompt_success_rate, "
            "then continuous_score, then minimize false_positive_rate. If none meet constraints, "
            "use recall-aware fallback score."
        ),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
