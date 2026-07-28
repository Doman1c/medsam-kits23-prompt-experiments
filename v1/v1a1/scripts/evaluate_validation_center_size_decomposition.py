from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.data import load_npz, normalize_image
from v1a1.inference import expand_box
from v1a1.utils import parse_int_list, resolve_device

SIDES = ("left", "right")
AREA_ORDER = ("00_10", "10_50", "50_90", "90_100")
MODES = (
    "auto_center_auto_size",
    "oracle_center_auto_size",
    "auto_center_oracle_size",
    "oracle_center_oracle_size",
)
LOCKED_EXPANSION = 1.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only MedSAM center-versus-size decomposition. "
            "All four modes use ground-truth presence."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--medsam-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--expansion-scale", type=float, default=LOCKED_EXPANSION)
    parser.add_argument("--expected-val-cases", type=int, default=20)
    parser.add_argument("--device", default="cuda")
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
    minimum = tensor.amin(dim=(2, 3), keepdim=True)
    maximum = tensor.amax(dim=(2, 3), keepdim=True)
    return (tensor - minimum) / (maximum - minimum).clamp_min(1e-8)


@torch.inference_mode()
def segment_with_box(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    box: tuple[float, float, float, float],
    original_h: int,
    original_w: int,
) -> np.ndarray:
    box_1024 = np.asarray(box, dtype=np.float32) / np.asarray(
        [original_w, original_h, original_w, original_h], dtype=np.float32
    ) * 1024.0
    box_tensor = torch.as_tensor(
        box_1024, dtype=torch.float32, device=embedding.device
    )[None, None]
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


def validate_lock(calibration_path: Path, expansion_scale: float) -> dict[str, Any]:
    config = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected = {
        "variant": "smooth",
        "max_gap": 0,
        "left_threshold": 0.20,
        "right_threshold": 0.30,
        "expansion_scale": 1.20,
    }
    for key, expected_value in expected.items():
        actual = config.get(key)
        if isinstance(expected_value, float):
            if abs(float(actual) - expected_value) > 1e-12:
                raise ValueError(f"Calibration mismatch for {key}: {actual}")
        elif actual != expected_value:
            raise ValueError(f"Calibration mismatch for {key}: {actual}")
    if abs(expansion_scale - LOCKED_EXPANSION) > 1e-12:
        raise ValueError(f"Expansion must remain {LOCKED_EXPANSION}")
    return config


def validate_inputs(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    expected_val_cases: int,
) -> pd.DataFrame:
    required = {
        "case_id", "slice_idx", "npz_path", "image_height", "image_width",
        "combined_area_bin", "combined_relative_area",
    }
    for side in SIDES:
        required.update({
            f"{side}_gt_presence", f"{side}_gt_x1", f"{side}_gt_y1",
            f"{side}_gt_x2", f"{side}_gt_y2",
            f"{side}_smooth_center_x_norm", f"{side}_smooth_center_y_norm",
            f"{side}_smooth_width_norm", f"{side}_smooth_height_norm",
        })
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {missing}")

    manifest = manifest.copy()
    predictions = predictions.copy()
    for frame in (manifest, predictions):
        frame["case_id"] = frame["case_id"].astype(str)
        frame["slice_idx"] = frame["slice_idx"].astype(int)

    val = manifest[manifest["split"].astype(str) == "val"].copy()
    if val["case_id"].nunique() != expected_val_cases:
        raise ValueError(
            f"Expected {expected_val_cases} validation cases, found {val['case_id'].nunique()}"
        )
    val_keys = set(zip(val["case_id"], val["slice_idx"]))
    pred_keys = set(zip(predictions["case_id"], predictions["slice_idx"]))
    if val_keys != pred_keys:
        raise ValueError(
            f"Validation/prediction key mismatch: missing={len(val_keys-pred_keys)}, "
            f"extra={len(pred_keys-val_keys)}"
        )
    if predictions.duplicated(["case_id", "slice_idx"]).any():
        raise ValueError("Duplicate prediction rows")

    positive = predictions[
        predictions["left_gt_presence"].astype(int).eq(1)
        | predictions["right_gt_presence"].astype(int).eq(1)
    ].copy()
    missing_npz = [p for p in positive["npz_path"].astype(str) if not Path(p).is_file()]
    if missing_npz:
        raise FileNotFoundError(f"Missing {len(missing_npz)} positive validation NPZ files")
    return positive.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)


def gt_center_size(row: pd.Series, side: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (
        float(row[f"{side}_gt_{key}"]) for key in ("x1", "y1", "x2", "y2")
    )
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        max(x2 - x1, 1.0),
        max(y2 - y1, 1.0),
    )


def auto_center_size(row: pd.Series, side: str) -> tuple[float, float, float, float]:
    image_width = float(row["image_width"])
    image_height = float(row["image_height"])
    return (
        float(row[f"{side}_smooth_center_x_norm"]) * image_width,
        float(row[f"{side}_smooth_center_y_norm"]) * image_height,
        max(float(row[f"{side}_smooth_width_norm"]) * image_width, 1.0),
        max(float(row[f"{side}_smooth_height_norm"]) * image_height, 1.0),
    )


def make_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    return (
        max(0.0, center_x - width / 2.0),
        max(0.0, center_y - height / 2.0),
        min(float(image_width), center_x + width / 2.0),
        min(float(image_height), center_y + height / 2.0),
    )


def boxes_for_mode(
    row: pd.Series,
    mode: str,
    expansion_scale: float,
) -> list[tuple[float, float, float, float]]:
    use_oracle_center = mode.startswith("oracle_center")
    use_oracle_size = mode.endswith("oracle_size")
    image_width = int(row["image_width"])
    image_height = int(row["image_height"])
    boxes = []
    for side in SIDES:
        if int(row[f"{side}_gt_presence"]) != 1:
            continue
        auto_cx, auto_cy, auto_w, auto_h = auto_center_size(row, side)
        gt_cx, gt_cy, gt_w, gt_h = gt_center_size(row, side)
        tight = make_box(
            gt_cx if use_oracle_center else auto_cx,
            gt_cy if use_oracle_center else auto_cy,
            gt_w if use_oracle_size else auto_w,
            gt_h if use_oracle_size else auto_h,
            image_width,
            image_height,
        )
        boxes.append(expand_box(tight, expansion_scale, image_width, image_height))
    return boxes


def build_wide(details: pd.DataFrame) -> pd.DataFrame:
    metadata = details[
        ["case_id", "slice_idx", "npz_path", "combined_area_bin", "combined_relative_area"]
    ].drop_duplicates(["case_id", "slice_idx"])
    dice = details.pivot(index=["case_id", "slice_idx"], columns="mode", values="dice")
    dice = dice.rename(columns={mode: f"{mode}_dice" for mode in MODES}).reset_index()
    result = metadata.merge(dice, on=["case_id", "slice_idx"], validate="one_to_one")
    m0 = result["auto_center_auto_size_dice"]
    m1 = result["oracle_center_auto_size_dice"]
    m2 = result["auto_center_oracle_size_dice"]
    m3 = result["oracle_center_oracle_size_dice"]
    result["oracle_center_gain"] = m1 - m0
    result["oracle_size_gain"] = m2 - m0
    result["joint_oracle_box_gain"] = m3 - m0
    result["remaining_size_gap_after_oracle_center"] = m3 - m1
    result["remaining_center_gap_after_oracle_size"] = m3 - m2
    result["center_size_interaction"] = m3 - m2 - m1 + m0
    return result


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    m0 = frame["auto_center_auto_size_dice"]
    m1 = frame["oracle_center_auto_size_dice"]
    m2 = frame["auto_center_oracle_size_dice"]
    m3 = frame["oracle_center_oracle_size_dice"]
    total = float((m3 - m0).mean())
    center = float((m1 - m0).mean())
    size = float((m2 - m0).mean())
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
        "center_size_interaction_mean": float((m3 - m2 - m1 + m0).mean()),
        "signed_center_gain_fraction_of_joint_gap": center / total if abs(total) > 1e-12 else float("nan"),
        "signed_size_gain_fraction_of_joint_gap": size / total if abs(total) > 1e-12 else float("nan"),
        "oracle_center_improves_rate": float((m1 > m0 + 1e-8).mean()),
        "oracle_center_worsens_rate": float((m1 < m0 - 1e-8).mean()),
        "oracle_size_improves_rate": float((m2 > m0 + 1e-8).mean()),
        "oracle_size_worsens_rate": float((m2 < m0 - 1e-8).mean()),
    }


def grouped_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, subset in frame.groupby(column, dropna=False):
        row = {column: value}
        row.update(summarize(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    predictions_path = Path(args.predictions)
    calibration_path = Path(args.calibration_json)
    checkpoint_path = Path(args.checkpoint)
    medsam_repo = Path(args.medsam_repo)
    output_dir = Path(args.output_dir)

    for path in (manifest_path, predictions_path, calibration_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not medsam_repo.is_dir():
        raise FileNotFoundError(medsam_repo)

    calibration = validate_lock(calibration_path, args.expansion_scale)
    positive = validate_inputs(
        pd.read_csv(manifest_path),
        pd.read_csv(predictions_path),
        args.expected_val_cases,
    )

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this validation experiment")
    print(f"Selected device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path)).to(device)
    model.eval()
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("MedSAM model is not on CUDA")

    labels = parse_int_list(args.target_labels)
    details: list[dict[str, Any]] = []
    started = time.time()

    for _, row in tqdm(
        positive.iterrows(),
        total=len(positive),
        desc="Validation center-size decomposition",
    ):
        image, label = load_npz(row["npz_path"])
        target = np.isin(label, np.asarray(labels, dtype=int)).astype(np.uint8)
        height, width = image.shape
        image_tensor = prepare_image(image, device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=True, dtype=torch.float16
        ):
            embedding = model.image_encoder(image_tensor)

        for mode in MODES:
            prediction = np.zeros_like(target, dtype=np.uint8)
            boxes = boxes_for_mode(row, mode, args.expansion_scale)
            for box in boxes:
                prediction |= segment_with_box(model, embedding, box, height, width)
            details.append({
                "case_id": str(row["case_id"]),
                "slice_idx": int(row["slice_idx"]),
                "npz_path": row["npz_path"],
                "combined_area_bin": row["combined_area_bin"],
                "combined_relative_area": float(row["combined_relative_area"]),
                "mode": mode,
                "num_prompts": len(boxes),
                "dice": dice_score(prediction, target),
                "iou": iou_score(prediction, target),
            })

    elapsed = time.time() - started
    detail_frame = pd.DataFrame(details)
    if len(detail_frame) != len(positive) * len(MODES):
        raise RuntimeError("Unexpected result row count")

    wide = build_wide(detail_frame)
    overall = summarize(wide)
    by_area = grouped_summary(wide, "combined_area_bin")
    by_area["_order"] = by_area["combined_area_bin"].map(
        {name: index for index, name in enumerate(AREA_ORDER)}
    )
    by_area = by_area.sort_values("_order").drop(columns="_order")
    by_case = grouped_summary(wide, "case_id").sort_values(
        "joint_oracle_box_gain_mean_M3_minus_M0", ascending=False
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "center_size_medsam_details.csv"
    per_slice_path = output_dir / "center_size_decomposition_per_slice.csv"
    detail_frame.to_csv(detail_path, index=False)
    wide.to_csv(per_slice_path, index=False)
    by_area.to_csv(output_dir / "center_size_decomposition_by_area.csv", index=False)
    by_case.to_csv(output_dir / "center_size_decomposition_per_case.csv", index=False)

    summary = {
        "status": "validation_only_center_versus_size_decomposition",
        "development_split": "validation",
        "test_set_used": False,
        "all_modes_use_ground_truth_presence": True,
        "modes": {
            "M0": "ground-truth presence + automatic smooth center + automatic smooth size",
            "M1": "ground-truth presence + oracle center + automatic smooth size",
            "M2": "ground-truth presence + automatic smooth center + oracle size",
            "M3": "ground-truth presence + oracle center + oracle size",
        },
        "locked_configuration": {
            "candidate": calibration.get("candidate"),
            "variant": calibration.get("variant"),
            "max_gap": calibration.get("max_gap"),
            "left_threshold": calibration.get("left_threshold"),
            "right_threshold": calibration.get("right_threshold"),
            "expansion_scale": args.expansion_scale,
            "target_labels": list(labels),
        },
        "data": {
            "validation_cases": int(positive["case_id"].nunique()),
            "validation_positive_slices": int(len(positive)),
            "result_rows": int(len(detail_frame)),
        },
        "overall_decomposition": overall,
        "by_area": by_area.to_dict(orient="records"),
        "top_joint_gap_cases": by_case.head(5).to_dict(orient="records"),
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_mib": float(torch.cuda.max_memory_allocated() / 1024**2),
        },
        "artifact_sha256": {
            "manifest": sha256(manifest_path),
            "predictions": sha256(predictions_path),
            "calibration_json": sha256(calibration_path),
            "medsam_checkpoint": sha256(checkpoint_path),
            "details": sha256(detail_path),
            "per_slice": sha256(per_slice_path),
        },
        "research_guardrail": (
            "This experiment uses validation only. Do not revisit the frozen test set "
            "during model or hyperparameter selection."
        ),
    }
    summary_path = output_dir / "center_size_decomposition_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved validation decomposition to: {output_dir}")


if __name__ == "__main__":
    main()
