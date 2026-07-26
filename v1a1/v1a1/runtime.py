from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .inference import decode_dual_outputs
from .losses import compute_dual_loss
from .metrics import evaluate_prediction_frame


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def decoded_batch_to_rows(
    batch: dict[str, Any],
    decoded: list[dict[str, Any]],
) -> list[dict]:
    rows: list[dict] = []
    batch_size = len(decoded)
    for i in range(batch_size):
        row = {
            "case_id": batch["case_id"][i],
            "slice_idx": int(batch["slice_idx"][i]),
            "npz_path": batch["npz_path"][i],
            "image_height": int(batch["original_height"][i]),
            "image_width": int(batch["original_width"][i]),
            "combined_relative_area": float(batch["combined_relative_area"][i]),
            "combined_area_bin": batch["combined_area_bin"][i],
        }
        for side_index, side in enumerate(("left", "right")):
            gt_present = int(batch["presence"][i, side_index] > 0.5)
            row[f"{side}_gt_presence"] = gt_present
            row[f"{side}_area_bin"] = batch[f"{side}_area_bin"][i]
            gt_box = batch["gt_boxes"][i, side_index].tolist()
            for key, value in zip(("x1", "y1", "x2", "y2"), gt_box):
                row[f"{side}_gt_{key}"] = float(value)
            prediction = decoded[i][side]
            row[f"{side}_raw_presence_probability"] = prediction["presence_probability"]
            row[f"{side}_raw_presence"] = prediction["presence"]
            row[f"{side}_raw_heatmap_peak_probability"] = prediction["heatmap_peak_probability"]
            row[f"{side}_raw_center_x_norm"] = prediction["center_x_norm"]
            row[f"{side}_raw_center_y_norm"] = prediction["center_y_norm"]
            row[f"{side}_raw_width_norm"] = prediction["width_norm"]
            row[f"{side}_raw_height_norm"] = prediction["height_norm"]
            for key, value in zip(("x1", "y1", "x2", "y2"), prediction["box"]):
                row[f"{side}_raw_{key}"] = float(value)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    expansion_scale: float = 1.15,
    presence_threshold: float = 0.5,
    use_amp: bool = True,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    loss_sums: defaultdict[str, float] = defaultdict(float)
    batches = 0
    rows: list[dict] = []
    for batch_cpu in loader:
        batch = move_batch_to_device(batch_cpu, device)
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
            dtype=torch.float16,
        ):
            outputs = model(batch["image"])
            _, loss_metrics = compute_dual_loss(outputs, batch)
        decoded = decode_dual_outputs(
            outputs,
            batch["original_width"],
            batch["original_height"],
            presence_threshold=presence_threshold,
        )
        rows.extend(decoded_batch_to_rows(batch_cpu, decoded))
        for key, value in loss_metrics.items():
            loss_sums[key] += float(value)
        batches += 1
    frame = pd.DataFrame(rows)
    prompt_summary, _ = evaluate_prediction_frame(
        frame,
        variant="raw",
        expansion_scale=expansion_scale,
        presence_threshold=presence_threshold,
    )
    summary = {f"val_{key}": value / max(batches, 1) for key, value in loss_sums.items()}
    summary.update({f"val_{key}": value for key, value in prompt_summary.items() if key not in {"variant", "expansion_scale"}})
    return summary, frame
