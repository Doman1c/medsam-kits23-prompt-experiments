from __future__ import annotations

from typing import Any

import numpy as np
import torch


def expand_box(
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    scale: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = max(x2 - x1, 1.0) * float(scale)
    height = max(y2 - y1, 1.0) * float(scale)
    return (
        max(0.0, cx - width / 2.0),
        max(0.0, cy - height / 2.0),
        min(float(image_width), cx + width / 2.0),
        min(float(image_height), cy + height / 2.0),
    )


def box_iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@torch.no_grad()
def decode_dual_outputs(
    outputs: dict[str, torch.Tensor],
    original_width: torch.Tensor,
    original_height: torch.Tensor,
    presence_threshold: float = 0.5,
    min_size_norm: float = 0.005,
    max_size_norm: float = 0.95,
) -> list[dict[str, Any]]:
    presence_probability = torch.sigmoid(outputs["presence_logits"])
    heatmap_probability = torch.sigmoid(outputs["heatmap_logits"])
    batch, sides, heatmap_height, heatmap_width = heatmap_probability.shape
    flat_index = heatmap_probability.view(batch, sides, -1).argmax(dim=-1)
    peak_y = torch.div(flat_index, heatmap_width, rounding_mode="floor")
    peak_x = flat_index % heatmap_width

    offset_map = torch.sigmoid(outputs["offset_logits"].view(batch, 2, 2, heatmap_height, heatmap_width))
    log_size_map = outputs["log_size_map"].view(batch, 2, 2, heatmap_height, heatmap_width)

    results: list[dict[str, Any]] = []
    side_names = ("left", "right")
    for b in range(batch):
        item: dict[str, Any] = {}
        for side_index, side in enumerate(side_names):
            x = int(peak_x[b, side_index])
            y = int(peak_y[b, side_index])
            offset = offset_map[b, side_index, :, y, x]
            log_size = log_size_map[b, side_index, :, y, x]
            center_x_norm = float((x + offset[0]) / heatmap_width)
            center_y_norm = float((y + offset[1]) / heatmap_height)
            size = log_size.exp().clamp(min_size_norm, max_size_norm)
            width_norm, height_norm = float(size[0]), float(size[1])
            image_width = int(original_width[b])
            image_height = int(original_height[b])
            cx = center_x_norm * image_width
            cy = center_y_norm * image_height
            box_width = width_norm * image_width
            box_height = height_norm * image_height
            box = (
                max(0.0, cx - box_width / 2.0),
                max(0.0, cy - box_height / 2.0),
                min(float(image_width), cx + box_width / 2.0),
                min(float(image_height), cy + box_height / 2.0),
            )
            probability = float(presence_probability[b, side_index])
            item[side] = {
                "presence_probability": probability,
                "presence": int(probability >= presence_threshold),
                "heatmap_peak_probability": float(heatmap_probability[b, side_index, y, x]),
                "center_x_norm": center_x_norm,
                "center_y_norm": center_y_norm,
                "width_norm": width_norm,
                "height_norm": height_norm,
                "box": box,
            }
        results.append(item)
    return results


def rolling_median(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    radius = window // 2
    output = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        output[i] = np.median(values[lo:hi], axis=0)
    return output


def smooth_case_side(
    probability: np.ndarray,
    centers: np.ndarray,
    sizes: np.ndarray,
    threshold: float = 0.5,
    max_gap: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability_smooth = rolling_median(probability[:, None], window=3)[:, 0]
    present = probability_smooth >= threshold

    # Remove isolated single-slice positives.
    for i in range(1, len(present) - 1):
        if present[i] and not present[i - 1] and not present[i + 1]:
            present[i] = False

    # Fill short gaps bounded by positives.
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

    center_smooth = rolling_median(centers, window=3)
    log_size_smooth = rolling_median(np.log(np.clip(sizes, 1e-5, None)), window=3)
    size_smooth = np.exp(log_size_smooth)
    return probability_smooth, present.astype(np.uint8), center_smooth, size_smooth
