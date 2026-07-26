from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.calibration import load_config
from v1a1.data import load_npz, normalize_image
from v1a1.inference import expand_box
from v1a1.utils import parse_float_list, parse_int_list, resolve_device, save_json


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator) if denominator else 1.0


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def prepare_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = normalize_image(image)
    tensor = torch.from_numpy(image)[None, None].float().to(device)
    tensor = tensor.repeat(1, 3, 1, 1)
    tensor = F.interpolate(tensor, size=(1024, 1024), mode="bicubic", align_corners=False)
    min_v = tensor.amin(dim=(2, 3), keepdim=True)
    max_v = tensor.amax(dim=(2, 3), keepdim=True)
    return (tensor - min_v) / (max_v - min_v).clamp_min(1e-8)


@torch.inference_mode()
def segment_with_box(model, embedding: torch.Tensor, box, original_h: int, original_w: int) -> np.ndarray:
    box_1024 = np.asarray(box, dtype=np.float32) / np.asarray(
        [original_w, original_h, original_w, original_h], dtype=np.float32
    ) * 1024.0
    box_tensor = torch.as_tensor(box_1024, dtype=torch.float32, device=embedding.device)[None, None]
    sparse, dense = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    prediction = torch.sigmoid(low_res_logits)
    prediction = F.interpolate(prediction, size=(original_h, original_w), mode="bilinear", align_corners=False)
    return (prediction[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def boxes_for_variant(row: pd.Series, variant: str, scale: float) -> list[tuple[float, float, float, float]]:
    boxes = []
    for side in ("left", "right"):
        if int(row[f"{side}_{variant}_presence"]) != 1:
            continue
        box = tuple(float(row[f"{side}_{variant}_{key}"]) for key in ("x1", "y1", "x2", "y2"))
        boxes.append(expand_box(box, scale, int(row.image_width), int(row.image_height)))
    return boxes


def oracle_boxes(row: pd.Series, scale: float | None) -> list[tuple[float, float, float, float]]:
    boxes = []
    for side in ("left", "right"):
        if int(row[f"{side}_gt_presence"]) != 1:
            continue
        box = tuple(float(row[f"{side}_gt_{key}"]) for key in ("x1", "y1", "x2", "y2"))
        if scale is not None:
            box = expand_box(box, scale, int(row.image_width), int(row.image_height))
        boxes.append(box)
    return boxes


def aggregate(frame: pd.DataFrame) -> dict:
    if len(frame) == 0:
        return {"n": 0, "dice_mean": 0.0, "dice_median": 0.0, "iou_mean": 0.0}
    return {
        "n": int(len(frame)),
        "dice_mean": float(frame.dice.mean()),
        "dice_median": float(frame.dice.median()),
        "iou_mean": float(frame.iou.mean()),
    }


def summarize(details: pd.DataFrame, variant: str, scale: float) -> dict:
    subset = details[(details.mode == "calibrated") & (details.expansion_scale == scale)]
    oracle_tight = details[details.mode == "oracle_tight"]
    oracle_scaled = details[(details.mode == "oracle_scaled") & (details.expansion_scale == scale)]
    by_area = {
        area_bin: aggregate(subset[subset.combined_area_bin == area_bin])
        for area_bin in ("00_10", "10_50", "50_90", "90_100")
    }
    by_case = subset.groupby("case_id").dice.mean()
    oracle_by_case = oracle_scaled.groupby("case_id").dice.mean()
    overall = aggregate(subset)
    oracle_overall = aggregate(oracle_scaled)
    return {
        "variant": variant,
        "expansion_scale": float(scale),
        "overall": overall,
        "case_level_dice_mean": float(by_case.mean()),
        "by_combined_area_bin": by_area,
        "oracle_tight_overall": aggregate(oracle_tight),
        "oracle_scaled_overall": oracle_overall,
        "oracle_scaled_case_level_dice_mean": float(oracle_by_case.mean()),
        "oracle_gap_overall": float(oracle_overall["dice_mean"] - overall["dice_mean"]),
        "oracle_gap_case_level": float(oracle_by_case.mean() - by_case.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen MedSAM validation sweep for one calibrated checkpoint")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", default="1.10,1.15,1.20,1.25")
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    medsam_repo = Path(args.medsam_repo)
    if not medsam_repo.exists():
        raise FileNotFoundError(medsam_repo)
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    config = load_config(args.calibration_json)
    scales = parse_float_list(args.scales)
    target_labels = parse_int_list(args.target_labels)
    frame = pd.read_csv(args.predictions)
    if not args.include_negatives:
        frame = frame[(frame.left_gt_presence == 1) | (frame.right_gt_presence == 1)]
    if args.max_samples is not None:
        frame = frame.head(args.max_samples)
    if len(frame) == 0:
        raise RuntimeError("No validation rows selected")

    device = resolve_device(args.device)
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
    model.eval()
    details = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="MedSAM validation sweep"):
        image, label = load_npz(row.npz_path)
        gt = np.isin(label, np.asarray(target_labels)).astype(np.uint8)
        height, width = image.shape
        image_tensor = prepare_image(image, device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
        ):
            embedding = model.image_encoder(image_tensor)

        mode_boxes: list[tuple[str, float, list]] = [("oracle_tight", 1.0, oracle_boxes(row, None))]
        for scale in scales:
            mode_boxes.append(("oracle_scaled", scale, oracle_boxes(row, scale)))
            mode_boxes.append(("calibrated", scale, boxes_for_variant(row, config.variant, scale)))
        for mode, scale, boxes in mode_boxes:
            prediction = np.zeros_like(gt, dtype=np.uint8)
            for box in boxes:
                prediction |= segment_with_box(model, embedding, box, height, width)
            details.append({
                "case_id": row.case_id,
                "slice_idx": int(row.slice_idx),
                "npz_path": row.npz_path,
                "combined_area_bin": row.combined_area_bin,
                "combined_relative_area": float(row.combined_relative_area),
                "mode": mode,
                "variant": config.variant,
                "expansion_scale": float(scale),
                "num_prompts": len(boxes),
                "dice": dice_score(prediction, gt),
                "iou": iou_score(prediction, gt),
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_frame = pd.DataFrame(details)
    detail_frame.to_csv(output_dir / "medsam_validation_details.csv", index=False)
    summaries = [summarize(detail_frame, config.variant, scale) for scale in scales]
    summary_table = pd.DataFrame([
        {
            "variant": row["variant"],
            "expansion_scale": row["expansion_scale"],
            "dice_mean": row["overall"]["dice_mean"],
            "dice_median": row["overall"]["dice_median"],
            "case_level_dice_mean": row["case_level_dice_mean"],
            "oracle_gap_overall": row["oracle_gap_overall"],
            "oracle_gap_case_level": row["oracle_gap_case_level"],
        }
        for row in summaries
    ]).sort_values(["case_level_dice_mean", "dice_mean"], ascending=False)
    summary_table.to_csv(output_dir / "medsam_validation_scale_table.csv", index=False)
    best_scale = float(summary_table.iloc[0].expansion_scale)
    best_summary = next(row for row in summaries if row["expansion_scale"] == best_scale)
    payload = {
        "calibration": config.to_dict(),
        "scale_summaries": summaries,
        "recommended_by_validation_medsam": best_summary,
        "selection_rule": "Maximize case-level mean Dice, break ties with slice-level mean Dice.",
    }
    save_json(output_dir / "medsam_validation_summary.json", payload)
    print(summary_table.to_string(index=False))
    print("\nRecommended scale from frozen MedSAM validation:")
    print(json.dumps(best_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
