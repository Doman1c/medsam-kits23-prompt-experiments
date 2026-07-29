from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

SIDES = ("left", "right")
AREA_ORDER = ("00_10", "10_50", "50_90", "90_100")
MODES = (
    "auto_center_auto_size",
    "oracle_center_auto_size",
    "auto_center_oracle_size",
    "oracle_center_oracle_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only center-versus-size decomposition. All modes use GT presence and the locked selected raw/smooth variant.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--final-config", required=True)
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--expected-val-cases", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("target label list is empty")
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
def segment_with_box(model: torch.nn.Module, embedding: torch.Tensor, box: tuple[float, float, float, float], original_h: int, original_w: int) -> np.ndarray:
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


def gt_center_size(row: pd.Series, side: str) -> tuple[float, float, float, float]:
    x1 = float(row[f"{side}_gt_x1"])
    y1 = float(row[f"{side}_gt_y1"])
    x2 = float(row[f"{side}_gt_x2"])
    y2 = float(row[f"{side}_gt_y2"])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, 1.0), max(y2 - y1, 1.0))


def auto_center_size(row: pd.Series, side: str, variant: str) -> tuple[float, float, float, float]:
    width = float(row["image_width"])
    height = float(row["image_height"])
    return (
        float(row[f"{side}_{variant}_center_x_norm"]) * width,
        float(row[f"{side}_{variant}_center_y_norm"]) * height,
        max(float(row[f"{side}_{variant}_width_norm"]) * width, 1.0),
        max(float(row[f"{side}_{variant}_height_norm"]) * height, 1.0),
    )


def box_from_center_size(center_x: float, center_y: float, width: float, height: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return (
        max(0.0, center_x - width / 2.0),
        max(0.0, center_y - height / 2.0),
        min(float(image_width), center_x + width / 2.0),
        min(float(image_height), center_y + height / 2.0),
    )


def boxes_for_mode(row: pd.Series, mode: str, variant: str, expansion_scale: float, expand_box) -> list[tuple[float, float, float, float]]:
    boxes = []
    image_width = int(row["image_width"])
    image_height = int(row["image_height"])
    oracle_center = mode.startswith("oracle_center")
    oracle_size = mode.endswith("oracle_size")
    for side in SIDES:
        if int(row[f"{side}_gt_presence"]) != 1:
            continue
        auto_cx, auto_cy, auto_w, auto_h = auto_center_size(row, side, variant)
        gt_cx, gt_cy, gt_w, gt_h = gt_center_size(row, side)
        tight = box_from_center_size(
            gt_cx if oracle_center else auto_cx,
            gt_cy if oracle_center else auto_cy,
            gt_w if oracle_size else auto_w,
            gt_h if oracle_size else auto_h,
            image_width,
            image_height,
        )
        boxes.append(expand_box(tight, expansion_scale, image_width, image_height))
    return boxes


def decomposition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    m0 = frame["auto_center_auto_size_dice"]
    m1 = frame["oracle_center_auto_size_dice"]
    m2 = frame["auto_center_oracle_size_dice"]
    m3 = frame["oracle_center_oracle_size_dice"]
    total = float((m3 - m0).mean())
    center = float((m1 - m0).mean())
    size = float((m2 - m0).mean())
    interaction = float((m3 - m2 - m1 + m0).mean())
    return {
        "n_slices": int(len(frame)),
        "M0_auto_center_auto_size_mean": float(m0.mean()),
        "M0_auto_center_auto_size_median": float(m0.median()),
        "M1_oracle_center_auto_size_mean": float(m1.mean()),
        "M1_oracle_center_auto_size_median": float(m1.median()),
        "M2_auto_center_oracle_size_mean": float(m2.mean()),
        "M2_auto_center_oracle_size_median": float(m2.median()),
        "M3_oracle_center_oracle_size_mean": float(m3.mean()),
        "M3_oracle_center_oracle_size_median": float(m3.median()),
        "oracle_center_gain_mean_M1_minus_M0": center,
        "oracle_size_gain_mean_M2_minus_M0": size,
        "joint_oracle_box_gain_mean_M3_minus_M0": total,
        "remaining_size_gap_mean_M3_minus_M1": float((m3 - m1).mean()),
        "remaining_center_gap_mean_M3_minus_M2": float((m3 - m2).mean()),
        "center_size_interaction_mean": interaction,
        "signed_center_gain_fraction_of_joint_gap": center / total if abs(total) > 1e-12 else float("nan"),
        "signed_size_gain_fraction_of_joint_gap": size / total if abs(total) > 1e-12 else float("nan"),
        "interpretation_note": "Center and size replacement effects are not assumed additive. Interaction = M3 - M2 - M1 + M0.",
    }


def grouped_summary(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_column, dropna=False):
        row = {group_column: group}
        row.update(decomposition_summary(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    v1a1_root = repo_root / "v1" / "v1a1"
    sys.path.insert(0, str(v1a1_root))
    from v1a1.data import load_npz, normalize_image
    from v1a1.inference import expand_box

    manifest_path = Path(args.manifest)
    prediction_path = Path(args.predictions)
    config_path = Path(args.final_config)
    medsam_repo = Path(args.medsam_repo)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    for path in (manifest_path, prediction_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not medsam_repo.is_dir():
        raise FileNotFoundError(medsam_repo)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("development_split") != "validation":
        raise ValueError("final config is not marked as validation-only")
    if config.get("test_set_used") is not False:
        raise ValueError("final config indicates test use")
    variant = str(config["variant"])
    if variant not in {"raw", "smooth"}:
        raise ValueError(f"unsupported variant: {variant}")
    expansion_scale = float(config["expansion_scale"])

    manifest = pd.read_csv(manifest_path)
    predictions = pd.read_csv(prediction_path)
    manifest["case_id"] = manifest["case_id"].astype(str)
    predictions["case_id"] = predictions["case_id"].astype(str)
    manifest["slice_idx"] = manifest["slice_idx"].astype(int)
    predictions["slice_idx"] = predictions["slice_idx"].astype(int)
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    if val["case_id"].nunique() != args.expected_val_cases:
        raise ValueError(f"expected {args.expected_val_cases} validation cases, found {val['case_id'].nunique()}")
    val_keys = set(zip(val["case_id"], val["slice_idx"]))
    pred_keys = set(zip(predictions["case_id"], predictions["slice_idx"]))
    if pred_keys != val_keys:
        raise ValueError(f"prediction/validation key mismatch: missing={len(val_keys - pred_keys)}, extra={len(pred_keys - val_keys)}")

    positive = predictions[
        predictions["left_gt_presence"].astype(int).eq(1)
        | predictions["right_gt_presence"].astype(int).eq(1)
    ].copy().sort_values(["case_id", "slice_idx"])

    required = {"case_id", "slice_idx", "npz_path", "image_height", "image_width", "combined_area_bin", "combined_relative_area"}
    for side in SIDES:
        required.update({
            f"{side}_gt_presence", f"{side}_gt_x1", f"{side}_gt_y1", f"{side}_gt_x2", f"{side}_gt_y2",
            f"{side}_{variant}_center_x_norm", f"{side}_{variant}_center_y_norm",
            f"{side}_{variant}_width_norm", f"{side}_{variant}_height_norm",
        })
    missing = sorted(required - set(positive.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path)).to(device)
    model.eval()

    target_labels = parse_int_list(args.target_labels)
    details = []
    started = time.time()
    for _, row in tqdm(positive.iterrows(), total=len(positive), desc=f"{args.experiment} validation center-size decomposition"):
        image, label = load_npz(row["npz_path"])
        gt = np.isin(label, np.asarray(target_labels, dtype=int)).astype(np.uint8)
        height, width = image.shape
        image_tensor = prepare_image(image, device, normalize_image)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            embedding = model.image_encoder(image_tensor)
        for mode in MODES:
            boxes = boxes_for_mode(row, mode, variant, expansion_scale, expand_box)
            prediction = np.zeros_like(gt, dtype=np.uint8)
            for box in boxes:
                prediction |= segment_with_box(model, embedding, box, original_h=height, original_w=width)
            details.append({
                "case_id": str(row["case_id"]),
                "slice_idx": int(row["slice_idx"]),
                "npz_path": row["npz_path"],
                "combined_area_bin": row["combined_area_bin"],
                "combined_relative_area": float(row["combined_relative_area"]),
                "mode": mode,
                "variant": variant,
                "expansion_scale": expansion_scale,
                "num_prompts": len(boxes),
                "dice": dice_score(prediction, gt),
                "iou": iou_score(prediction, gt),
            })

    detail_frame = pd.DataFrame(details)
    expected = len(positive) * len(MODES)
    if len(detail_frame) != expected:
        raise RuntimeError(f"expected {expected} detail rows, found {len(detail_frame)}")

    metadata = detail_frame[["case_id", "slice_idx", "npz_path", "combined_area_bin", "combined_relative_area"]].drop_duplicates(["case_id", "slice_idx"])
    dice_wide = detail_frame.pivot(index=["case_id", "slice_idx"], columns="mode", values="dice").reset_index().rename(columns={mode: f"{mode}_dice" for mode in MODES})
    iou_wide = detail_frame.pivot(index=["case_id", "slice_idx"], columns="mode", values="iou").reset_index().rename(columns={mode: f"{mode}_iou" for mode in MODES})
    per_slice = metadata.merge(dice_wide, on=["case_id", "slice_idx"], validate="one_to_one").merge(iou_wide, on=["case_id", "slice_idx"], validate="one_to_one")

    m0 = per_slice["auto_center_auto_size_dice"]
    m1 = per_slice["oracle_center_auto_size_dice"]
    m2 = per_slice["auto_center_oracle_size_dice"]
    m3 = per_slice["oracle_center_oracle_size_dice"]
    per_slice["oracle_center_gain"] = m1 - m0
    per_slice["oracle_size_gain"] = m2 - m0
    per_slice["joint_oracle_box_gain"] = m3 - m0
    per_slice["center_size_interaction"] = m3 - m2 - m1 + m0

    overall = decomposition_summary(per_slice)
    by_area = grouped_summary(per_slice, "combined_area_bin")
    by_area["_order"] = by_area["combined_area_bin"].map({name: i for i, name in enumerate(AREA_ORDER)})
    by_area = by_area.sort_values("_order").drop(columns="_order")
    by_case = grouped_summary(per_slice, "case_id").sort_values("joint_oracle_box_gain_mean_M3_minus_M0", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_frame.to_csv(output_dir / "center_size_medsam_details.csv", index=False)
    per_slice.to_csv(output_dir / "center_size_decomposition_per_slice.csv", index=False)
    by_area.to_csv(output_dir / "center_size_decomposition_by_area.csv", index=False)
    by_case.to_csv(output_dir / "center_size_decomposition_per_case.csv", index=False)

    summary = {
        "status": f"{args.experiment}_validation_center_size_decomposition",
        "experiment": args.experiment,
        "development_split": "validation",
        "test_set_used": False,
        "all_modes_use_ground_truth_presence": True,
        "selected_variant": variant,
        "expansion_scale": expansion_scale,
        "data": {
            "validation_cases": int(positive["case_id"].nunique()),
            "positive_slices": int(len(positive)),
            "result_rows": int(len(detail_frame)),
        },
        "overall_decomposition": overall,
        "by_area": by_area.to_dict(orient="records"),
        "top_joint_gap_cases": by_case.head(5).to_dict(orient="records"),
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "elapsed_seconds": time.time() - started,
        },
        "artifact_sha256": {
            "manifest": sha256(manifest_path),
            "predictions": sha256(prediction_path),
            "final_config": sha256(config_path),
            "medsam_checkpoint": sha256(checkpoint_path),
        },
        "research_guardrail": "Validation-only diagnostic. Do not use the frozen test set to select V1a.2 variants.",
    }
    (output_dir / "center_size_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels = ["M0\nAuto C + Auto S", "M1\nOracle C + Auto S", "M2\nAuto C + Oracle S", "M3\nOracle C + Oracle S"]
    values = [
        overall["M0_auto_center_auto_size_mean"],
        overall["M1_oracle_center_auto_size_mean"],
        overall["M2_auto_center_oracle_size_mean"],
        overall["M3_oracle_center_oracle_size_mean"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean MedSAM Dice")
    ax.set_title(f"{args.experiment} validation center-size decomposition")
    for i, value in enumerate(values):
        ax.text(i, value + 0.012, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "center_size_mode_means.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
