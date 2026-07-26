from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "image" not in data or "label" not in data:
            raise KeyError(f"{path} must contain image and label")
        image = np.asarray(data["image"], dtype=np.float32)
        label = np.asarray(data["label"], dtype=np.int16)
    if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
        raise ValueError(f"Invalid arrays in {path}: image={image.shape}, label={label.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite image values in {path}")
    return image, label


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_v, max_v = float(image.min()), float(image.max())
    if min_v < -1e-4 or max_v > 1.0001:
        image = (image - min_v) / max(max_v - min_v, 1e-8)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def read_manifest(path: str | Path, split: str | None = None) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    if not rows:
        raise RuntimeError(f"No rows found in {path} for split={split!r}")
    return rows


def gaussian_heatmap(
    height: int,
    width: int,
    center_x_norm: float,
    center_y_norm: float,
    sigma: float,
) -> tuple[np.ndarray, int, int, float, float]:
    fx = float(np.clip(center_x_norm * width, 0.0, width - 1e-6))
    fy = float(np.clip(center_y_norm * height, 0.0, height - 1e-6))
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    offset_x = fx - ix
    offset_y = fy - iy
    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2.0 * sigma**2))
    heatmap[iy, ix] = 1.0
    return heatmap.astype(np.float32), ix, iy, offset_x, offset_y


class DualKidneyDataset(Dataset):
    SIDES = ("left", "right")

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        input_size: int = 256,
        heatmap_size: int = 64,
        input_mode: str = "2.5d",
        augment: bool = False,
        heatmap_sigma: float = 2.0,
    ) -> None:
        self.rows = read_manifest(manifest, split)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.input_mode = input_mode.lower()
        self.augment = bool(augment)
        self.heatmap_sigma = float(heatmap_sigma)
        if self.input_mode not in {"2d", "2.5d"}:
            raise ValueError("input_mode must be '2d' or '2.5d'")
        self.index_by_case_slice = {
            (row["case_id"], int(row["slice_idx"])): i
            for i, row in enumerate(self.rows)
        }

    def __len__(self) -> int:
        return len(self.rows)

    def _path_for_offset(self, row: dict[str, str], offset: int) -> str:
        if self.input_mode == "2d" or offset == 0:
            return row["npz_path"]
        target_slice = int(row["slice_idx"]) + offset
        index = self.index_by_case_slice.get((row["case_id"], target_slice))
        if index is not None:
            return self.rows[index]["npz_path"]
        # Overfit subsets may omit neighboring manifest rows; recover them from the case folder.
        sibling = Path(row["npz_path"]).with_name(f"slice_{target_slice:04d}.npz")
        return str(sibling) if sibling.exists() else row["npz_path"]

    def _resize(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(normalize_image(image))[None, None]
        return F.interpolate(
            tensor,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        images: list[torch.Tensor] = []
        original_shape: tuple[int, int] | None = None
        for offset in (-1, 0, 1):
            image, _ = load_npz(self._path_for_offset(row, offset))
            images.append(self._resize(image))
            if offset == 0:
                original_shape = image.shape
        assert original_shape is not None
        if self.input_mode == "2d":
            image_tensor = images[1][None]
        else:
            image_tensor = torch.stack(images, dim=0)

        # Keep anatomical left/right channels fixed. Do not horizontally flip.
        if self.augment:
            if random.random() < 0.85:
                scale = random.uniform(0.90, 1.10)
                shift = random.uniform(-0.05, 0.05)
                image_tensor = (image_tensor * scale + shift).clamp(0.0, 1.0)
            if random.random() < 0.30:
                noise = torch.randn_like(image_tensor) * random.uniform(0.0, 0.02)
                image_tensor = (image_tensor + noise).clamp(0.0, 1.0)

        presence = torch.zeros(2, dtype=torch.float32)
        centers = torch.zeros((2, 2), dtype=torch.float32)
        center_indices = torch.zeros((2, 2), dtype=torch.long)  # x, y
        offsets = torch.zeros((2, 2), dtype=torch.float32)
        sizes = torch.zeros((2, 2), dtype=torch.float32)
        log_sizes = torch.zeros((2, 2), dtype=torch.float32)
        relative_areas = torch.zeros(2, dtype=torch.float32)
        gt_boxes = torch.zeros((2, 4), dtype=torch.float32)
        heatmaps = torch.zeros((2, self.heatmap_size, self.heatmap_size), dtype=torch.float32)
        area_bins: list[str] = []

        for side_index, side in enumerate(self.SIDES):
            present = int(row[f"{side}_presence"])
            presence[side_index] = float(present)
            area_bins.append(row[f"{side}_area_bin"])
            if not present:
                continue
            cx = float(row[f"{side}_center_x_norm"])
            cy = float(row[f"{side}_center_y_norm"])
            width_norm = max(float(row[f"{side}_box_width_norm"]), 1e-5)
            height_norm = max(float(row[f"{side}_box_height_norm"]), 1e-5)
            heatmap, ix, iy, ox, oy = gaussian_heatmap(
                self.heatmap_size,
                self.heatmap_size,
                cx,
                cy,
                sigma=self.heatmap_sigma,
            )
            heatmaps[side_index] = torch.from_numpy(heatmap)
            centers[side_index] = torch.tensor([cx, cy])
            center_indices[side_index] = torch.tensor([ix, iy])
            offsets[side_index] = torch.tensor([ox, oy])
            sizes[side_index] = torch.tensor([width_norm, height_norm])
            log_sizes[side_index] = torch.log(sizes[side_index])
            relative_areas[side_index] = float(row[f"{side}_relative_area"])
            gt_boxes[side_index] = torch.tensor(
                [float(row[f"{side}_{key}"]) for key in ("x1", "y1", "x2", "y2")]
            )

        return {
            "image": image_tensor.float(),
            "presence": presence,
            "heatmap": heatmaps,
            "center": centers,
            "center_index": center_indices,
            "offset": offsets,
            "size": sizes,
            "log_size": log_sizes,
            "relative_area": relative_areas,
            "gt_boxes": gt_boxes,
            "original_height": torch.tensor(original_shape[0], dtype=torch.long),
            "original_width": torch.tensor(original_shape[1], dtype=torch.long),
            "case_id": row["case_id"],
            "slice_idx": torch.tensor(int(row["slice_idx"]), dtype=torch.long),
            "npz_path": row["npz_path"],
            "left_area_bin": area_bins[0],
            "right_area_bin": area_bins[1],
            "combined_relative_area": torch.tensor(float(row["combined_relative_area"]), dtype=torch.float32),
            "combined_area_bin": row["combined_area_bin"],
        }


def build_dual_balanced_sampler(
    dataset: DualKidneyDataset,
    negative_ratio: float = 0.4,
    near_negative_fraction: float = 0.7,
    seed: int = 42,
) -> WeightedRandomSampler:
    side_bin_counts: Counter[tuple[str, str]] = Counter()
    positive_rows: list[int] = []
    near_negative: list[int] = []
    far_negative: list[int] = []
    for i, row in enumerate(dataset.rows):
        present_sides = [side for side in dataset.SIDES if int(row[f"{side}_presence"])]
        if present_sides:
            positive_rows.append(i)
            for side in present_sides:
                side_bin_counts[(side, row[f"{side}_area_bin"])] += 1
        elif int(row.get("is_near_negative", "0")):
            near_negative.append(i)
        else:
            far_negative.append(i)

    if not positive_rows:
        raise RuntimeError("No positive rows in training split")

    weights = np.zeros(len(dataset), dtype=np.float64)
    for i in positive_rows:
        row = dataset.rows[i]
        contributions = []
        for side in dataset.SIDES:
            if int(row[f"{side}_presence"]):
                key = (side, row[f"{side}_area_bin"])
                contributions.append(1.0 / max(side_bin_counts[key], 1))
        weights[i] = float(np.mean(contributions)) if contributions else 0.0

    positive_mass = weights[positive_rows].sum()
    if positive_mass <= 0:
        raise RuntimeError("Positive sampling weights are zero")
    weights[positive_rows] /= positive_mass

    negative_mass = max(float(negative_ratio), 0.0)
    near_mass = negative_mass * near_negative_fraction
    far_mass = negative_mass * (1.0 - near_negative_fraction)
    if near_negative:
        weights[near_negative] = near_mass / len(near_negative)
    if far_negative:
        weights[far_negative] = far_mass / len(far_negative)
    if not near_negative and far_negative:
        weights[far_negative] = negative_mass / len(far_negative)
    if not far_negative and near_negative:
        weights[near_negative] = negative_mass / len(near_negative)

    num_samples = int(round(len(positive_rows) * (1.0 + negative_ratio)))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=max(num_samples, 1),
        replacement=True,
        generator=generator,
    )
