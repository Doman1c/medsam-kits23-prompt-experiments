from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Target-label list is empty")
    return values


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def prepare_image(image: np.ndarray, device: torch.device, normalize_image) -> torch.Tensor:
    image = normalize_image(image)
    tensor = torch.from_numpy(image)[None, None].float().to(device)
    tensor = tensor.repeat(1, 3, 1, 1)
    tensor = F.interpolate(tensor, size=(1024, 1024), mode="bicubic", align_corners=False)
    minimum = tensor.amin(dim=(2, 3), keepdim=True)
    maximum = tensor.amax(dim=(2, 3), keepdim=True)
    return (tensor - minimum) / (maximum - minimum).clamp_min(1e-8)


@torch.inference_mode()
def segment_with_box(model, embedding: torch.Tensor, box: tuple[float, float, float, float], original_h: int, original_w: int) -> np.ndarray:
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
    probability = torch.sigmoid(low_res_logits)
    probability = F.interpolate(probability, size=(original_h, original_w), mode="bilinear", align_corners=False)
    return (probability[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def boxes_for_candidate(row: pd.Series, variant: str, expansion_scale: float, expand_box) -> list[tuple[float, float, float, float]]:
    boxes = []
    image_width = int(row["image_width"])
    image_height = int(row["image_height"])
    for side in ("left", "right"):
        if int(row[f"{side}_{variant}_presence"]) != 1:
            continue
        tight = tuple(float(row[f"{side}_{variant}_{key}"]) for key in ("x1", "y1", "x2", "y2"))
        boxes.append(expand_box(tight, expansion_scale, image_width, image_height))
    return boxes


def summarize_candidate(frame: pd.DataFrame) -> dict:
    positive = frame[frame["gt_positive"]]
    negative = frame[~frame["gt_positive"]]
    case_slice = frame.groupby("case_id", sort=False)["dice"].mean()
    positive_case_slice = positive.groupby("case_id", sort=False)["dice"].mean()

    case_volume_rows = []
    for case_id, subset in frame.groupby("case_id", sort=False):
        tp = int(subset["tp_pixels"].sum())
        fp = int(subset["fp_pixels"].sum())
        fn = int(subset["fn_pixels"].sum())
        denominator = 2 * tp + fp + fn
        volume_dice = 1.0 if denominator == 0 else 2.0 * tp / denominator
        case_volume_rows.append({"case_id": case_id, "volume_dice": volume_dice, "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn})
    case_volume = pd.DataFrame(case_volume_rows)
    negative_count = len(negative)
    negative_false_mask_rate = float(negative["pred_positive"].mean()) if negative_count else float("nan")
    negative_specificity = 1.0 - negative_false_mask_rate if negative_count else float("nan")

    return {
        "n_slices": int(len(frame)),
        "n_positive_slices": int(len(positive)),
        "n_negative_slices": int(len(negative)),
        "all_slice_dice_mean": float(frame["dice"].mean()),
        "all_slice_dice_median": float(frame["dice"].median()),
        "all_slice_iou_mean": float(frame["iou"].mean()),
        "all_slice_case_mean_dice": float(case_slice.mean()),
        "positive_slice_dice_mean": float(positive["dice"].mean()),
        "positive_slice_dice_median": float(positive["dice"].median()),
        "positive_slice_case_mean_dice": float(positive_case_slice.mean()),
        "positive_slice_detection_recall": float(positive["pred_positive"].mean()),
        "negative_slice_specificity": negative_specificity,
        "negative_false_positive_mask_rate": negative_false_mask_rate,
        "negative_prompt_rate": float((negative["num_prompts"] > 0).mean()),
        "negative_predicted_fraction_mean": float(negative["predicted_fraction"].mean()),
        "negative_predicted_fraction_when_positive_mean": float(
            negative.loc[negative["pred_positive"], "predicted_fraction"].mean()
        ) if negative["pred_positive"].any() else 0.0,
        "case_volume_dice_mean": float(case_volume["volume_dice"].mean()),
        "case_volume_dice_median": float(case_volume["volume_dice"].median()),
        "total_tp_pixels": int(frame["tp_pixels"].sum()),
        "total_fp_pixels": int(frame["fp_pixels"].sum()),
        "total_fn_pixels": int(frame["fn_pixels"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple calibrated prompt candidates on all validation slices, including fully negative slices. The MedSAM image embedding is reused across candidates.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root / "v1" / "v1a1"))
    from v1a1.calibration import load_config
    from v1a1.data import load_npz, normalize_image
    from v1a1.inference import expand_box

    manifest_payload = json.loads(Path(args.candidate_manifest).read_text(encoding="utf-8"))
    candidate_specs = manifest_payload["candidates"]
    if not candidate_specs:
        raise ValueError("No candidates supplied")

    candidates = []
    reference_keys = None
    for spec in candidate_specs:
        prediction_path = Path(spec["predictions"])
        calibration_path = Path(spec["calibration_json"])
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(calibration_path)
        frame = pd.read_csv(prediction_path)
        frame["case_id"] = frame["case_id"].astype(str)
        frame["slice_idx"] = frame["slice_idx"].astype(int)
        frame = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
        keys = list(zip(frame["case_id"], frame["slice_idx"]))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"Key mismatch for candidate {spec['name']}")
        config = load_config(calibration_path)
        candidates.append({
            "name": str(spec["name"]),
            "frame": frame,
            "variant": config.variant,
            "expansion_scale": float(spec["expansion_scale"]),
            "predictions": str(prediction_path),
            "calibration_json": str(calibration_path),
        })

    first = candidates[0]["frame"]
    medsam_repo = Path(args.medsam_repo)
    checkpoint = Path(args.checkpoint)
    if not medsam_repo.is_dir():
        raise FileNotFoundError(medsam_repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    model.eval()

    target_labels = parse_int_list(args.target_labels)
    details = []
    started = time.time()
    for index in tqdm(range(len(first)), desc="Negative-aware all-slice MedSAM"):
        reference_row = first.iloc[index]
        image, label = load_npz(reference_row["npz_path"])
        gt = np.isin(label, np.asarray(target_labels, dtype=int)).astype(np.uint8)
        height, width = image.shape
        image_tensor = prepare_image(image, device, normalize_image)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            embedding = model.image_encoder(image_tensor)
        gt_bool = gt.astype(bool)
        gt_pixels = int(gt_bool.sum())

        for candidate in candidates:
            row = candidate["frame"].iloc[index]
            boxes = boxes_for_candidate(row, candidate["variant"], candidate["expansion_scale"], expand_box)
            prediction = np.zeros_like(gt, dtype=np.uint8)
            for box in boxes:
                prediction |= segment_with_box(model, embedding, box, height, width)
            pred_bool = prediction.astype(bool)
            tp = int(np.logical_and(pred_bool, gt_bool).sum())
            fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
            fn = int(np.logical_and(~pred_bool, gt_bool).sum())
            pred_pixels = int(pred_bool.sum())
            details.append({
                "candidate": candidate["name"],
                "case_id": str(row["case_id"]),
                "slice_idx": int(row["slice_idx"]),
                "npz_path": row["npz_path"],
                "combined_area_bin": row.get("combined_area_bin", None),
                "combined_relative_area": float(row.get("combined_relative_area", 0.0)),
                "variant": candidate["variant"],
                "expansion_scale": candidate["expansion_scale"],
                "gt_positive": bool(gt_pixels > 0),
                "pred_positive": bool(pred_pixels > 0),
                "num_prompts": int(len(boxes)),
                "gt_pixels": gt_pixels,
                "predicted_pixels": pred_pixels,
                "predicted_fraction": pred_pixels / float(height * width),
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "dice": dice_score(prediction, gt),
                "iou": iou_score(prediction, gt),
            })

    details_frame = pd.DataFrame(details)
    summaries = []
    for candidate in candidates:
        subset = details_frame[details_frame["candidate"] == candidate["name"]].copy()
        summary = summarize_candidate(subset)
        summary.update({
            "candidate": candidate["name"],
            "variant": candidate["variant"],
            "expansion_scale": candidate["expansion_scale"],
            "predictions": candidate["predictions"],
            "calibration_json": candidate["calibration_json"],
        })
        summaries.append(summary)

    summary_table = pd.DataFrame(summaries).sort_values(
        ["case_volume_dice_mean", "all_slice_dice_mean", "negative_false_positive_mask_rate"],
        ascending=[False, False, True],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_frame.to_csv(output_dir / "negative_aware_medsam_details.csv", index=False)
    summary_table.to_csv(output_dir / "negative_aware_candidate_comparison.csv", index=False)
    payload = {
        "status": "validation_only_negative_aware_medsam",
        "development_split": "validation",
        "test_set_used": False,
        "selection_rule_for_description": "Rank by case-volume Dice, then all-slice mean Dice, then lower negative false-positive mask rate.",
        "candidates": summaries,
        "ranked_candidates": summary_table["candidate"].tolist(),
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    (output_dir / "negative_aware_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_table.to_string(index=False))
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
