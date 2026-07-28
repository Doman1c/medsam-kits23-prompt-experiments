from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.data import load_npz, normalize_image  # noqa: E402
from v1a1.inference import expand_box  # noqa: E402
from v1a1.utils import parse_int_list, resolve_device  # noqa: E402

SIDES = ("left", "right")
AREA_ORDER = ("00_10", "10_50", "50_90", "90_100")
LOCKED = {
    "epoch": 6,
    "weights": "ema",
    "variant": "smooth",
    "max_gap": 0,
    "left_threshold": 0.20,
    "right_threshold": 0.30,
    "expansion_scale": 1.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-test A/B/C decomposition: A=automatic presence+box, "
            "B=oracle presence+automatic box, C=oracle presence+oracle box."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--medsam-details", required=True)
    parser.add_argument("--frozen-test-summary", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    parser.add_argument("--expected-positive-slices", type=int, default=1473)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(
        2.0 * np.logical_and(prediction, target).sum() / denominator
    )


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(
        np.logical_and(prediction, target).sum() / union
    )


def validate_lock(summary_path: Path, calibration_path: Path, expansion: float) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    frozen = summary.get("locked_configuration", {})
    for key, expected in LOCKED.items():
        actual = frozen.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"Frozen {key}: expected {expected}, got {actual}")
        elif actual != expected:
            raise ValueError(f"Frozen {key}: expected {expected}, got {actual}")
    for key in ("variant", "max_gap", "left_threshold", "right_threshold", "expansion_scale"):
        expected = LOCKED[key]
        actual = calibration.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"Calibration {key}: expected {expected}, got {actual}")
        elif actual != expected:
            raise ValueError(f"Calibration {key}: expected {expected}, got {actual}")
    if abs(float(expansion) - LOCKED["expansion_scale"]) > 1e-12:
        raise ValueError("Expansion must remain locked at 1.20")
    return summary


def prepare_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(normalize_image(image))[None, None].float().to(device)
    tensor = F.interpolate(
        tensor.repeat(1, 3, 1, 1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    low = tensor.amin(dim=(2, 3), keepdim=True)
    high = tensor.amax(dim=(2, 3), keepdim=True)
    return (tensor - low) / (high - low).clamp_min(1e-8)


@torch.inference_mode()
def segment_with_box(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    box: tuple[float, float, float, float],
    height: int,
    width: int,
) -> np.ndarray:
    scaled = np.asarray(box, dtype=np.float32) / np.asarray(
        [width, height, width, height], dtype=np.float32
    ) * 1024.0
    box_tensor = torch.as_tensor(
        scaled, dtype=torch.float32, device=embedding.device
    )[None, None]
    sparse, dense = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    prediction = F.interpolate(
        torch.sigmoid(logits),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return (prediction[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def existing_scores(details: pd.DataFrame, expected: int) -> pd.DataFrame:
    required = {
        "case_id", "slice_idx", "mode", "dice", "iou", "num_prompts",
        "combined_area_bin", "combined_relative_area",
    }
    missing = sorted(required - set(details.columns))
    if missing:
        raise ValueError(f"MedSAM details missing columns: {missing}")
    details = details.copy()
    details["case_id"] = details["case_id"].astype(str)
    details["slice_idx"] = details["slice_idx"].astype(int)
    frames = {}
    for mode, prefix in (("dual_smooth", "A_automatic"), ("oracle_scaled", "C_oracle_box")):
        frame = details[details["mode"] == mode].copy()
        if len(frame) != expected:
            raise ValueError(f"{mode}: expected {expected}, found {len(frame)}")
        frames[mode] = frame[
            ["case_id", "slice_idx", "dice", "iou", "num_prompts",
             "combined_area_bin", "combined_relative_area"]
        ].rename(columns={
            "dice": f"{prefix}_dice",
            "iou": f"{prefix}_iou",
            "num_prompts": f"{prefix}_num_prompts",
        })
    return frames["dual_smooth"].merge(
        frames["oracle_scaled"]\
            [["case_id", "slice_idx", "C_oracle_box_dice", "C_oracle_box_iou", "C_oracle_box_num_prompts"]],
        on=["case_id", "slice_idx"],
        validate="one_to_one",
    )


def add_presence_status(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["gt_prompt_count"] = sum(output[f"{s}_gt_presence"].astype(int) for s in SIDES)
    output["pred_prompt_count"] = sum(output[f"{s}_smooth_presence"].astype(int) for s in SIDES)
    output["false_positive_sides"] = sum(
        (output[f"{s}_gt_presence"].astype(int).eq(0)
         & output[f"{s}_smooth_presence"].astype(int).eq(1)).astype(int)
        for s in SIDES
    )
    output["false_negative_sides"] = sum(
        (output[f"{s}_gt_presence"].astype(int).eq(1)
         & output[f"{s}_smooth_presence"].astype(int).eq(0)).astype(int)
        for s in SIDES
    )
    exact = output["false_positive_sides"].eq(0) & output["false_negative_sides"].eq(0)
    extra = output["false_positive_sides"].gt(0) & output["false_negative_sides"].eq(0)
    missing = output["false_positive_sides"].eq(0) & output["false_negative_sides"].gt(0)
    output["side_status"] = np.select(
        [exact, extra, missing],
        ["exact_sides", "extra_side", "missing_side"],
        default="mixed_side_error",
    )
    return output


def auto_boxes_for_gt_sides(row: pd.Series, expansion: float) -> list[tuple[float, float, float, float]]:
    boxes = []
    for side in SIDES:
        if int(row[f"{side}_gt_presence"]) != 1:
            continue
        box = tuple(float(row[f"{side}_smooth_{key}"]) for key in ("x1", "y1", "x2", "y2"))
        boxes.append(expand_box(
            box, expansion, int(row["image_width"]), int(row["image_height"])
        ))
    return boxes


def run_b(
    model: torch.nn.Module,
    device: torch.device,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    target_labels: list[int],
    expansion: float,
) -> pd.DataFrame:
    table = add_presence_status(scores.merge(
        predictions, on=["case_id", "slice_idx"], validate="one_to_one"
    ))
    rows: list[dict[str, Any]] = []
    exact = table[table["side_status"] == "exact_sides"]
    for _, row in exact.iterrows():
        rows.append({
            "case_id": str(row["case_id"]),
            "slice_idx": int(row["slice_idx"]),
            "side_status": row["side_status"],
            "gt_prompt_count": int(row["gt_prompt_count"]),
            "pred_prompt_count": int(row["pred_prompt_count"]),
            "B_oracle_presence_auto_box_num_prompts": int(row["gt_prompt_count"]),
            "B_oracle_presence_auto_box_dice": float(row["A_automatic_dice"]),
            "B_oracle_presence_auto_box_iou": float(row["A_automatic_iou"]),
            "B_result_source": "reused_A_exact_presence",
        })
    mismatch = table[table["side_status"] != "exact_sides"]
    for _, row in tqdm(mismatch.iterrows(), total=len(mismatch), desc="Oracle presence + automatic box"):
        image, label = load_npz(row["npz_path"])
        gt = np.isin(label, np.asarray(target_labels, dtype=int)).astype(np.uint8)
        height, width = image.shape
        image_tensor = prepare_image(image, device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16,
        ):
            embedding = model.image_encoder(image_tensor)
        prediction = np.zeros_like(gt, dtype=np.uint8)
        boxes = auto_boxes_for_gt_sides(row, expansion)
        for box in boxes:
            prediction |= segment_with_box(model, embedding, box, height, width)
        rows.append({
            "case_id": str(row["case_id"]),
            "slice_idx": int(row["slice_idx"]),
            "side_status": row["side_status"],
            "gt_prompt_count": int(row["gt_prompt_count"]),
            "pred_prompt_count": int(row["pred_prompt_count"]),
            "B_oracle_presence_auto_box_num_prompts": len(boxes),
            "B_oracle_presence_auto_box_dice": dice_score(prediction, gt),
            "B_oracle_presence_auto_box_iou": iou_score(prediction, gt),
            "B_result_source": "recomputed_presence_mismatch",
        })
    result = pd.DataFrame(rows)
    if len(result) != len(table) or result.duplicated(["case_id", "slice_idx"]).any():
        raise RuntimeError("B-mode row-count or duplicate-key failure")
    return result


def aggregate(frame: pd.DataFrame) -> dict[str, float | int]:
    total = float(frame["total_oracle_gap_dice"].mean())
    presence = float(frame["presence_decision_gain_dice"].mean())
    box = float(frame["box_localization_gap_dice"].mean())
    positive_total = np.clip(frame["total_oracle_gap_dice"].to_numpy(float), 0.0, None)
    positive_presence = np.minimum(
        np.clip(frame["presence_decision_gain_dice"].to_numpy(float), 0.0, None),
        positive_total,
    )
    positive_sum = float(positive_total.sum())
    recovered = float(positive_presence.sum())
    return {
        "n_slices": int(len(frame)),
        "A_automatic_dice_mean": float(frame["A_automatic_dice"].mean()),
        "A_automatic_dice_median": float(frame["A_automatic_dice"].median()),
        "B_oracle_presence_auto_box_dice_mean": float(frame["B_oracle_presence_auto_box_dice"].mean()),
        "B_oracle_presence_auto_box_dice_median": float(frame["B_oracle_presence_auto_box_dice"].median()),
        "C_oracle_box_dice_mean": float(frame["C_oracle_box_dice"].mean()),
        "C_oracle_box_dice_median": float(frame["C_oracle_box_dice"].median()),
        "presence_decision_gain_mean": presence,
        "presence_decision_gain_median": float(frame["presence_decision_gain_dice"].median()),
        "box_localization_gap_mean": box,
        "box_localization_gap_median": float(frame["box_localization_gap_dice"].median()),
        "total_oracle_gap_mean": total,
        "total_oracle_gap_median": float(frame["total_oracle_gap_dice"].median()),
        "signed_presence_fraction_of_total_gap": presence / total if abs(total) > 1e-12 else float("nan"),
        "signed_box_fraction_of_total_gap": box / total if abs(total) > 1e-12 else float("nan"),
        "positive_oracle_gap_sum": positive_sum,
        "capped_positive_presence_recovery_sum": recovered,
        "capped_positive_presence_recovery_fraction": recovered / positive_sum if positive_sum > 0 else float("nan"),
        "presence_oracle_improves_rate": float((frame["presence_decision_gain_dice"] > 1e-8).mean()),
        "presence_oracle_worsens_rate": float((frame["presence_decision_gain_dice"] < -1e-8).mean()),
        "presence_oracle_no_change_rate": float((frame["presence_decision_gain_dice"].abs() <= 1e-8).mean()),
    }


def grouped(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    records = []
    for value, subset in frame.groupby(column, dropna=False):
        record = {column: value}
        record.update(aggregate(subset))
        records.append(record)
    return pd.DataFrame(records)
