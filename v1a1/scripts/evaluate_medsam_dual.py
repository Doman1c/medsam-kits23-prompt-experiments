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

from v1a1.data import load_npz, normalize_image
from v1a1.inference import expand_box
from v1a1.utils import parse_int_list, resolve_device, save_json


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
    prediction = F.interpolate(
        prediction, size=(original_h, original_w), mode="bilinear", align_corners=False
    )
    return (prediction[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def boxes_for_mode(row: pd.Series, mode: str, scale: float) -> list[tuple[float, float, float, float]]:
    boxes = []
    for side in ("left", "right"):
        if mode.startswith("oracle"):
            present = int(row[f"{side}_gt_presence"])
            prefix = f"{side}_gt"
        else:
            variant = "raw" if mode == "dual_raw" else "smooth"
            present = int(row[f"{side}_{variant}_presence"])
            prefix = f"{side}_{variant}"
        if not present:
            continue
        box = tuple(float(row[f"{prefix}_{key}"]) for key in ("x1", "y1", "x2", "y2"))
        if mode in {"oracle_scaled", "dual_raw", "dual_smooth"}:
            box = expand_box(box, scale, int(row.image_width), int(row.image_height))
        boxes.append(box)
    return boxes


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "dice_mean": 0.0, "dice_median": 0.0, "iou_mean": 0.0}
    return {
        "n": len(rows),
        "dice_mean": float(np.mean([row["dice"] for row in rows])),
        "dice_median": float(np.median([row["dice"] for row in rows])),
        "iou_mean": float(np.mean([row["iou"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen MedSAM evaluation for dual left/right boxes")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--expansion-scale", type=float, default=1.15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    medsam_repo = Path(args.medsam_repo)
    if not medsam_repo.exists():
        raise FileNotFoundError(medsam_repo)
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    device = resolve_device(args.device)
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
    model.eval()
    target_labels = parse_int_list(args.target_labels)
    frame = pd.read_csv(args.predictions)
    if not args.include_negatives:
        frame = frame[(frame.left_gt_presence == 1) | (frame.right_gt_presence == 1)]
    if args.max_samples is not None:
        frame = frame.head(args.max_samples)
    if len(frame) == 0:
        raise RuntimeError("No rows selected for MedSAM evaluation")

    details = []
    modes = ("oracle_tight", "oracle_scaled", "dual_raw", "dual_smooth")
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Frozen MedSAM dual evaluation"):
        image, label = load_npz(row.npz_path)
        gt = np.isin(label, np.asarray(target_labels)).astype(np.uint8)
        height, width = image.shape
        image_tensor = prepare_image(image, device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16,
        ):
            embedding = model.image_encoder(image_tensor)
        for mode in modes:
            boxes = boxes_for_mode(row, mode, args.expansion_scale)
            prediction = np.zeros_like(gt, dtype=np.uint8)
            for box in boxes:
                prediction |= segment_with_box(model, embedding, box, height, width)
            details.append(
                {
                    "case_id": row.case_id,
                    "slice_idx": int(row.slice_idx),
                    "npz_path": row.npz_path,
                    "combined_area_bin": row.combined_area_bin,
                    "combined_relative_area": float(row.combined_relative_area),
                    "mode": mode,
                    "num_prompts": len(boxes),
                    "dice": dice_score(prediction, gt),
                    "iou": iou_score(prediction, gt),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_frame = pd.DataFrame(details)
    detail_frame.to_csv(output_dir / "medsam_dual_details.csv", index=False)
    summary = {}
    for mode in modes:
        mode_rows = [row for row in details if row["mode"] == mode]
        by_area = {
            area_bin: aggregate([row for row in mode_rows if row["combined_area_bin"] == area_bin])
            for area_bin in ("00_10", "10_50", "50_90", "90_100")
        }
        by_case: defaultdict[str, list[float]] = defaultdict(list)
        for row in mode_rows:
            by_case[row["case_id"]].append(row["dice"])
        summary[mode] = {
            "overall": aggregate(mode_rows),
            "by_combined_area_bin": by_area,
            "case_level_dice_mean": float(np.mean([np.mean(values) for values in by_case.values()])),
        }
    oracle = summary["oracle_scaled"]["overall"]["dice_mean"]
    summary["oracle_gap"] = {
        "dual_raw": float(oracle - summary["dual_raw"]["overall"]["dice_mean"]),
        "dual_smooth": float(oracle - summary["dual_smooth"]["overall"]["dice_mean"]),
    }
    summary["expansion_scale"] = args.expansion_scale
    save_json(output_dir / "medsam_dual_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
