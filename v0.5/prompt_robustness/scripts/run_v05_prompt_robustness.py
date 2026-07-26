from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry


@dataclass
class MedSAMOutput:
    mask: np.ndarray
    predicted_iou: float


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(
        2.0 * intersection / denominator
    )


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(intersection / union)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    labels, count = ndi.label(
        binary,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return binary

    areas = ndi.sum(binary, labels, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)


def prepare_target(
    label_map: np.ndarray,
    target_id: int,
    component_mode: str,
) -> np.ndarray:
    target = (label_map == int(target_id)).astype(np.uint8)

    if component_mode == "largest":
        return largest_connected_component(target)
    if component_mode == "full":
        return target

    raise ValueError(f"Unknown component mode: {component_mode}")


def mask_to_box(mask: np.ndarray, padding: int = 2) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot create a box from an empty mask.")

    height, width = mask.shape
    return np.asarray(
        [
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(width - 1, int(xs.max()) + padding),
            min(height - 1, int(ys.max()) + padding),
        ],
        dtype=np.float32,
    )


def sanitize_box(
    box: np.ndarray,
    image_shape: tuple[int, int],
    minimum_size: float = 2.0,
) -> np.ndarray:
    height, width = image_shape
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).tolist()

    x1 = float(np.clip(x1, 0.0, width - 1.0))
    x2 = float(np.clip(x2, 0.0, width - 1.0))
    y1 = float(np.clip(y1, 0.0, height - 1.0))
    y2 = float(np.clip(y2, 0.0, height - 1.0))

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if x2 - x1 < minimum_size:
        center_x = (x1 + x2) / 2.0
        x1 = center_x - minimum_size / 2.0
        x2 = center_x + minimum_size / 2.0

    if y2 - y1 < minimum_size:
        center_y = (y1 + y2) / 2.0
        y1 = center_y - minimum_size / 2.0
        y2 = center_y + minimum_size / 2.0

    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if x2 > width - 1:
        shift = x2 - (width - 1)
        x1 -= shift
        x2 = width - 1.0

    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if y2 > height - 1:
        shift = y2 - (height - 1)
        y1 -= shift
        y2 = height - 1.0

    return np.asarray(
        [
            np.clip(x1, 0.0, width - 1.0),
            np.clip(y1, 0.0, height - 1.0),
            np.clip(x2, 0.0, width - 1.0),
            np.clip(y2, 0.0, height - 1.0),
        ],
        dtype=np.float32,
    )


def scale_box(
    box: np.ndarray,
    scale: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Box scale must be greater than zero.")

    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(2.0, float(x2 - x1)) * scale
    height = max(2.0, float(y2 - y1)) * scale

    scaled = np.asarray(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        dtype=np.float32,
    )
    return sanitize_box(scaled, image_shape)


def shift_box(
    box: np.ndarray,
    level: float,
    image_shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Translate the whole box while preserving its size when possible."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    box_width = max(2.0, float(x2 - x1))
    box_height = max(2.0, float(y2 - y1))

    dx = float(rng.uniform(-level, level) * box_width)
    dy = float(rng.uniform(-level, level) * box_height)

    shifted = np.asarray([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    height, width = image_shape

    if shifted[0] < 0:
        shifted[[0, 2]] -= shifted[0]
    if shifted[2] > width - 1:
        shifted[[0, 2]] -= shifted[2] - (width - 1)
    if shifted[1] < 0:
        shifted[[1, 3]] -= shifted[1]
    if shifted[3] > height - 1:
        shifted[[1, 3]] -= shifted[3] - (height - 1)

    return sanitize_box(shifted, image_shape)


def corner_jitter_box(
    box: np.ndarray,
    level: float,
    image_shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    box_width = max(2.0, float(x2 - x1))
    box_height = max(2.0, float(y2 - y1))

    jittered = np.asarray(
        [
            x1 + rng.uniform(-level, level) * box_width,
            y1 + rng.uniform(-level, level) * box_height,
            x2 + rng.uniform(-level, level) * box_width,
            y2 + rng.uniform(-level, level) * box_height,
        ],
        dtype=np.float32,
    )
    return sanitize_box(jittered, image_shape)


def continuous_box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=np.float32)

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = continuous_box_area(box_a) + continuous_box_area(box_b) - intersection
    return 1.0 if union <= 0 else float(intersection / union)


def rasterized_box_bounds(
    box: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)

    left = int(np.clip(math.floor(x1), 0, width - 1))
    top = int(np.clip(math.floor(y1), 0, height - 1))
    right = int(np.clip(math.ceil(x2), left, width - 1))
    bottom = int(np.clip(math.ceil(y2), top, height - 1))
    return left, top, right, bottom


def box_quality_metrics(
    prompt_box: np.ndarray,
    ground_truth_box: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    image_shape = target.shape
    left, top, right, bottom = rasterized_box_bounds(
        prompt_box,
        image_shape,
    )

    inside = np.zeros_like(target, dtype=bool)
    inside[top : bottom + 1, left : right + 1] = True

    target_bool = target > 0
    target_pixels = int(target_bool.sum())
    target_pixels_in_box = int(np.logical_and(target_bool, inside).sum())
    box_pixels = int(inside.sum())

    coverage = (
        1.0
        if target_pixels == 0
        else float(target_pixels_in_box / target_pixels)
    )
    fill_ratio = (
        0.0
        if box_pixels == 0
        else float(target_pixels_in_box / box_pixels)
    )

    px1, py1, px2, py2 = np.asarray(prompt_box, dtype=np.float32)
    gx1, gy1, gx2, gy2 = np.asarray(ground_truth_box, dtype=np.float32)
    prompt_center = np.asarray([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
    gt_center = np.asarray([(gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0])
    gt_diagonal = math.hypot(max(1.0, gx2 - gx1), max(1.0, gy2 - gy1))
    center_error = float(np.linalg.norm(prompt_center - gt_center) / gt_diagonal)

    gt_area = continuous_box_area(ground_truth_box)
    prompt_area = continuous_box_area(prompt_box)
    area_ratio = 1.0 if gt_area <= 0 else float(prompt_area / gt_area)
    abs_log_area_ratio = float(abs(math.log(max(area_ratio, 1e-8))))

    return {
        "box_iou": box_iou(prompt_box, ground_truth_box),
        "target_coverage": coverage,
        "center_error_norm": center_error,
        "box_area_ratio": area_ratio,
        "abs_log_area_ratio": abs_log_area_ratio,
        "target_pixels_in_box": target_pixels_in_box,
        "target_fill_ratio": fill_ratio,
    }


class BoxPromptMedSAM:
    image_size = 1024

    def __init__(self, checkpoint: str, device: str | None = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("MedSAM device:", self.device)

    @torch.inference_mode()
    def encode_image(
        self,
        image: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got {image.shape}")

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        minimum, maximum = float(image.min()), float(image.max())
        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image)

        original_shape = (int(image.shape[0]), int(image.shape[1]))
        tensor = torch.from_numpy(image)[None, None].to(self.device)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        ).repeat(1, 3, 1, 1)

        embedding = self.model.image_encoder(tensor)
        return embedding, original_shape

    def scale_box(
        self,
        box: np.ndarray,
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        height, width = original_shape
        scaled = np.asarray(box, dtype=np.float32).copy()
        scaled[[0, 2]] *= self.image_size / width
        scaled[[1, 3]] *= self.image_size / height
        return torch.from_numpy(scaled)[None].to(self.device)

    @torch.inference_mode()
    def decode_box(
        self,
        image_embedding: torch.Tensor,
        original_shape: tuple[int, int],
        box: np.ndarray,
    ) -> MedSAMOutput:
        box_tensor = self.scale_box(box, original_shape)

        sparse, dense = self.model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None,
        )
        low_res_masks, predicted_iou = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = self.model.postprocess_masks(
            low_res_masks,
            input_size=(self.image_size, self.image_size),
            original_size=original_shape,
        )
        mask = (
            torch.sigmoid(logits)[0, 0] > 0.5
        ).cpu().numpy().astype(np.uint8)

        return MedSAMOutput(
            mask=mask,
            predicted_iou=float(predicted_iou[0, 0].cpu().item()),
        )


def load_split(
    split_path: Path,
    requested_splits: Iterable[str],
) -> tuple[dict, set[str]]:
    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)

    selected_cases: set[str] = set()
    for split_name in requested_splits:
        if split_name not in split:
            raise KeyError(
                f"Split '{split_name}' is missing from {split_path}."
            )
        selected_cases.update(str(case_id) for case_id in split[split_name])

    return split, selected_cases


def add_evaluated_areas(
    data: pd.DataFrame,
    processed_root: Path,
    component_mode: str,
    minimum_pixels: int,
) -> pd.DataFrame:
    records: list[dict] = []

    for row in tqdm(
        data.itertuples(index=False),
        total=len(data),
        desc="Inspecting target areas",
    ):
        sample_path = processed_root / row.npz_path
        if not sample_path.exists():
            continue

        try:
            with np.load(sample_path) as sample:
                label_map = sample["label"].astype(np.uint8)

            target = prepare_target(
                label_map,
                target_id=int(row.target_id),
                component_mode=component_mode,
            )
            evaluated_area = int(target.sum())
            if evaluated_area < minimum_pixels:
                continue

            record = row._asdict()
            record["evaluated_target_pixels"] = evaluated_area
            records.append(record)
        except Exception as error:
            print(
                f"[skip area] {row.case_id}, slice {row.slice_index}: "
                f"{type(error).__name__}: {error}"
            )

    if not records:
        raise RuntimeError("No valid slice-target samples remain.")

    result = pd.DataFrame(records)
    result["case_max_target_pixels"] = (
        result.groupby("case_id")["evaluated_target_pixels"]
        .transform("max")
        .astype(int)
    )
    result["relative_area_to_case_max"] = (
        result["evaluated_target_pixels"]
        / result["case_max_target_pixels"].clip(lower=1)
    )
    return result


def select_representative_slices(
    data: pd.DataFrame,
    samples_per_case: int,
) -> pd.DataFrame:
    if samples_per_case <= 0:
        raise ValueError("--samples-per-case must be greater than zero.")

    selected_frames: list[pd.DataFrame] = []

    for _, frame in data.groupby("case_id", sort=True):
        ordered = frame.sort_values(
            ["evaluated_target_pixels", "slice_index"]
        ).reset_index(drop=True)

        count = min(samples_per_case, len(ordered))
        positions = np.rint(
            np.linspace(0, len(ordered) - 1, count)
        ).astype(int)
        positions = np.unique(positions)

        selected = ordered.iloc[positions].copy()
        selected["selection_position"] = np.arange(len(selected))
        selected["selection_fraction"] = (
            0.0
            if len(selected) == 1
            else np.linspace(0.0, 1.0, len(selected))
        )
        selected_frames.append(selected)

    result = pd.concat(selected_frames, ignore_index=True)
    return result.sort_values(
        ["case_id", "slice_index"]
    ).reset_index(drop=True)


def assign_area_groups(selected: pd.DataFrame) -> pd.DataFrame:
    result = selected.copy()
    lower = float(result["evaluated_target_pixels"].quantile(1.0 / 3.0))
    upper = float(result["evaluated_target_pixels"].quantile(2.0 / 3.0))

    def group(area: float) -> str:
        if area <= lower:
            return "small"
        if area <= upper:
            return "medium"
        return "large"

    result["mask_area_group"] = result["evaluated_target_pixels"].map(group)
    return result


def build_conditions(
    scale_levels: list[float],
    shift_levels: list[float],
    jitter_levels: list[float],
    repeats: int,
) -> list[dict]:
    conditions: list[dict] = [
        {
            "perturbation_type": "oracle",
            "perturbation_level": 0.0,
            "repeat_index": 0,
        }
    ]

    for level in scale_levels:
        conditions.append(
            {
                "perturbation_type": "scale",
                "perturbation_level": float(level),
                "repeat_index": 0,
            }
        )

    for perturbation_type, levels in [
        ("shift", shift_levels),
        ("corner_jitter", jitter_levels),
    ]:
        for level in levels:
            for repeat_index in range(repeats):
                conditions.append(
                    {
                        "perturbation_type": perturbation_type,
                        "perturbation_level": float(level),
                        "repeat_index": repeat_index,
                    }
                )

    return conditions


def condition_seed(
    master_seed: int,
    sample_number: int,
    condition_number: int,
    repeat_index: int,
) -> int:
    return int(
        master_seed
        + 1_000_003 * sample_number
        + 10_007 * condition_number
        + 101 * repeat_index
    )


def make_prompt_box(
    ground_truth_box: np.ndarray,
    image_shape: tuple[int, int],
    condition: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbation_type = condition["perturbation_type"]
    level = float(condition["perturbation_level"])

    if perturbation_type == "oracle":
        return ground_truth_box.copy()
    if perturbation_type == "scale":
        return scale_box(ground_truth_box, level, image_shape)
    if perturbation_type == "shift":
        return shift_box(ground_truth_box, level, image_shape, rng)
    if perturbation_type == "corner_jitter":
        return corner_jitter_box(
            ground_truth_box,
            level,
            image_shape,
            rng,
        )

    raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def box_columns(prefix: str, box: np.ndarray) -> dict[str, float]:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return {
        f"{prefix}_x1": float(x1),
        f"{prefix}_y1": float(y1),
        f"{prefix}_x2": float(x2),
        f"{prefix}_y2": float(y2),
    }


def save_sample_visualization(
    output_path: Path,
    image: np.ndarray,
    target: np.ndarray,
    ground_truth_box: np.ndarray,
    oracle_prediction: np.ndarray,
    worst_box: np.ndarray,
    worst_prediction: np.ndarray,
    worst_title: str,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    for box in [ground_truth_box, worst_box]:
        x1, y1, x2, y2 = box
        axes[0].add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
            )
        )
    axes[0].set_title("Image: GT box + worst box")

    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("Evaluation target")

    axes[2].imshow(oracle_prediction, cmap="gray")
    axes[2].set_title("Oracle box prediction")

    axes[3].imshow(worst_prediction, cmap="gray")
    axes[3].set_title(worst_title)

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_intermediate_results(
    rows: list[dict],
    output_dir: Path,
) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "sample_results.csv",
            index=False,
        )


def create_summaries(results: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [
        "box_iou",
        "target_coverage",
        "center_error_norm",
        "box_area_ratio",
        "dice",
        "iou",
        "delta_dice",
        "dice_retention",
        "failure",
        "prompt_induced_failure",
        "relative_failure",
        "severe_drop",
    ]

    summary = (
        results.groupby(
            [
                "split_group",
                "perturbation_type",
                "perturbation_level",
            ],
            dropna=False,
        )[metric_columns]
        .agg(["mean", "std", "median", "min", "max", "count"])
    )
    summary.to_csv(output_dir / "summary_by_condition.csv")

    case_results = (
        results.groupby(
            [
                "case_id",
                "split_group",
                "perturbation_type",
                "perturbation_level",
            ],
            dropna=False,
        )[[
            "dice",
            "iou",
            "delta_dice",
            "dice_retention",
            "failure",
            "prompt_induced_failure",
            "relative_failure",
            "severe_drop",
        ]]
        .mean()
        .reset_index()
    )
    case_results.to_csv(output_dir / "case_results.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V0.5: evaluate MedSAM robustness to imperfect bounding-box "
            "prompts using fixed case splits and representative slices."
        )
    )
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-name",
        choices=["kidney", "tumor", "cyst"],
        default="kidney",
    )
    parser.add_argument(
        "--component-mode",
        choices=["largest", "full"],
        default="largest",
    )
    parser.add_argument("--samples-per-case", type=int, default=5)
    parser.add_argument("--min-evaluated-pixels", type=int, default=20)
    parser.add_argument("--box-padding", type=int, default=2)
    parser.add_argument(
        "--scale-levels",
        type=float,
        nargs="+",
        default=[0.75, 0.90, 1.10, 1.25, 1.50, 2.00],
    )
    parser.add_argument(
        "--shift-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
    )
    parser.add_argument(
        "--jitter-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-dice", type=float, default=0.80)
    parser.add_argument(
        "--relative-failure-retention",
        type=float,
        default=0.90,
        help="A prompt is a relative failure when Dice retention is below this.",
    )
    parser.add_argument(
        "--severe-drop-threshold",
        type=float,
        default=0.10,
        help="A severe drop means perturbed Dice decreases by at least this value.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Development limit after split filtering; 0 uses all cases.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Development limit after representative selection; 0 uses all.",
    )
    parser.add_argument("--save-visualizations", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.split_file.exists():
        raise FileNotFoundError(f"Split file not found: {args.split_file}")

    index_path = args.processed_root / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found: {index_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = args.output_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)

    data = pd.read_csv(index_path)
    required_columns = {
        "case_id",
        "slice_index",
        "npz_path",
        "target_id",
        "target_name",
        "target_pixels",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"index.csv is missing columns: {sorted(missing)}"
        )

    split_info, selected_case_ids = load_split(
        args.split_file,
        args.splits,
    )

    data = data[
        (data["target_name"] == args.target_name)
        & (data["case_id"].astype(str).isin(selected_case_ids))
    ].copy()

    if data.empty:
        raise RuntimeError("No indexed samples match the requested split.")

    if args.max_cases > 0:
        limited_cases = (
            data["case_id"].astype(str).drop_duplicates()
            .sort_values().head(args.max_cases).tolist()
        )
        data = data[data["case_id"].astype(str).isin(limited_cases)].copy()

    data = add_evaluated_areas(
        data,
        processed_root=args.processed_root,
        component_mode=args.component_mode,
        minimum_pixels=args.min_evaluated_pixels,
    )
    selected = select_representative_slices(
        data,
        samples_per_case=args.samples_per_case,
    )
    selected = assign_area_groups(selected)

    if args.max_samples > 0:
        selected = selected.head(args.max_samples).copy()

    split_lookup = {
        str(case_id): split_name
        for split_name in ["train", "val", "test"]
        for case_id in split_info.get(split_name, [])
    }
    selected["split_group"] = selected["case_id"].astype(str).map(split_lookup)
    selected.to_csv(
        args.output_dir / "selected_samples.csv",
        index=False,
    )

    conditions = build_conditions(
        scale_levels=args.scale_levels,
        shift_levels=args.shift_levels,
        jitter_levels=args.jitter_levels,
        repeats=args.repeats,
    )

    config = {
        "processed_root": str(args.processed_root),
        "checkpoint": str(args.checkpoint),
        "split_file": str(args.split_file),
        "splits": args.splits,
        "target_name": args.target_name,
        "component_mode": args.component_mode,
        "samples_per_case": args.samples_per_case,
        "min_evaluated_pixels": args.min_evaluated_pixels,
        "box_padding": args.box_padding,
        "scale_levels": args.scale_levels,
        "shift_levels": args.shift_levels,
        "jitter_levels": args.jitter_levels,
        "repeats": args.repeats,
        "seed": args.seed,
        "failure_dice": args.failure_dice,
        "relative_failure_retention": args.relative_failure_retention,
        "severe_drop_threshold": args.severe_drop_threshold,
        "selected_cases": int(selected["case_id"].nunique()),
        "selected_samples": int(len(selected)),
        "conditions_per_sample": len(conditions),
        "planned_decoder_calls": int(len(selected) * len(conditions)),
    }
    with (args.output_dir / "config_used.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print("Requested splits:", args.splits)
    print("Selected cases:", selected["case_id"].nunique())
    print("Selected samples:", len(selected))
    print("Conditions per sample:", len(conditions))
    print("Planned decoder calls:", len(selected) * len(conditions))

    medsam = BoxPromptMedSAM(
        str(args.checkpoint),
        device=args.device,
    )

    rows: list[dict] = []
    skipped = 0
    visualizations_saved = 0

    progress = tqdm(
        enumerate(selected.itertuples(index=False), start=1),
        total=len(selected),
        desc="Running V0.5 MedSAM",
    )

    for sample_number, row in progress:
        sample_path = args.processed_root / row.npz_path

        try:
            with np.load(sample_path) as sample:
                image = sample["image"].astype(np.float32)
                label_map = sample["label"].astype(np.uint8)

            target = prepare_target(
                label_map,
                target_id=int(row.target_id),
                component_mode=args.component_mode,
            )
            if int(target.sum()) < args.min_evaluated_pixels:
                raise ValueError("Target is too small after component selection.")

            ground_truth_box = mask_to_box(
                target,
                padding=args.box_padding,
            )
            image_embedding, original_shape = medsam.encode_image(image)

            oracle_output = medsam.decode_box(
                image_embedding,
                original_shape,
                ground_truth_box,
            )
            oracle_dice = dice_score(oracle_output.mask, target)
            oracle_iou = iou_score(oracle_output.mask, target)

            sample_predictions: list[dict] = []

            for condition_number, condition in enumerate(conditions):
                seed = condition_seed(
                    args.seed,
                    sample_number,
                    condition_number,
                    int(condition["repeat_index"]),
                )
                rng = np.random.default_rng(seed)

                prompt_box = make_prompt_box(
                    ground_truth_box,
                    image_shape=target.shape,
                    condition=condition,
                    rng=rng,
                )

                if condition["perturbation_type"] == "oracle":
                    output = oracle_output
                else:
                    output = medsam.decode_box(
                        image_embedding,
                        original_shape,
                        prompt_box,
                    )

                dice = dice_score(output.mask, target)
                mask_iou = iou_score(output.mask, target)
                retention = (
                    1.0
                    if oracle_dice <= 0
                    else float(dice / oracle_dice)
                )
                quality = box_quality_metrics(
                    prompt_box,
                    ground_truth_box,
                    target,
                )

                result = {
                    "case_id": str(row.case_id),
                    "slice_index": int(row.slice_index),
                    "npz_path": str(row.npz_path),
                    "split_group": str(row.split_group),
                    "target_id": int(row.target_id),
                    "target_name": str(row.target_name),
                    "component_mode": args.component_mode,
                    "mask_area": int(target.sum()),
                    "mask_area_group": str(row.mask_area_group),
                    "selection_position": int(row.selection_position),
                    "selection_fraction": float(row.selection_fraction),
                    "perturbation_type": condition["perturbation_type"],
                    "perturbation_level": float(
                        condition["perturbation_level"]
                    ),
                    "repeat_index": int(condition["repeat_index"]),
                    "repeat_seed": seed,
                    **box_columns("gt_box", ground_truth_box),
                    **box_columns("prompt_box", prompt_box),
                    **quality,
                    "oracle_dice": oracle_dice,
                    "oracle_iou": oracle_iou,
                    "oracle_medsam_predicted_iou": (
                        oracle_output.predicted_iou
                    ),
                    "dice": dice,
                    "iou": mask_iou,
                    "medsam_predicted_iou": output.predicted_iou,
                    "delta_dice": float(dice - oracle_dice),
                    "dice_retention": retention,
                    "failure": int(dice < args.failure_dice),
                    "prompt_induced_failure": int(
                        oracle_dice >= args.failure_dice
                        and dice < args.failure_dice
                    ),
                    "relative_failure": int(
                        retention < args.relative_failure_retention
                    ),
                    "severe_drop": int(
                        (dice - oracle_dice)
                        <= -args.severe_drop_threshold
                    ),
                }
                rows.append(result)
                sample_predictions.append(
                    {
                        "condition": condition,
                        "box": prompt_box,
                        "prediction": output.mask,
                        "dice": dice,
                    }
                )

            if visualizations_saved < args.save_visualizations:
                worst = min(
                    sample_predictions,
                    key=lambda item: item["dice"],
                )
                condition = worst["condition"]
                worst_title = (
                    f"{condition['perturbation_type']} "
                    f"{condition['perturbation_level']:.2f}, "
                    f"Dice={worst['dice']:.3f}"
                )
                save_sample_visualization(
                    visualization_dir
                    / (
                        f"{row.case_id}_z{int(row.slice_index):04d}_"
                        f"{row.target_name}.png"
                    ),
                    image=image,
                    target=target,
                    ground_truth_box=ground_truth_box,
                    oracle_prediction=oracle_output.mask,
                    worst_box=worst["box"],
                    worst_prediction=worst["prediction"],
                    worst_title=worst_title,
                )
                visualizations_saved += 1

            if args.save_every > 0 and sample_number % args.save_every == 0:
                save_intermediate_results(rows, args.output_dir)

        except Exception as error:
            print(
                f"[skip] {row.case_id}, slice {row.slice_index}: "
                f"{type(error).__name__}: {error}"
            )
            skipped += 1

    save_intermediate_results(rows, args.output_dir)

    if not rows:
        raise RuntimeError("No V0.5 samples were evaluated successfully.")

    results = pd.DataFrame(rows)
    create_summaries(results, args.output_dir)

    print("\nMean Dice by perturbation type:")
    print(
        results.groupby("perturbation_type")["dice"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nEvaluated cases:", results["case_id"].nunique())
    print(
        "Evaluated slice-target samples:",
        len(
            results[
                ["case_id", "slice_index", "target_name"]
            ].drop_duplicates()
        ),
    )
    print("Completed prompt evaluations:", len(results))
    print("Skipped samples:", skipped)
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
