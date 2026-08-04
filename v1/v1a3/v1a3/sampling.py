from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

SIDES = ("left", "right")
NEGATIVE_CATEGORIES = (
    "negative_mined",
    "negative_boundary",
    "negative_near",
    "negative_far",
)


@dataclass(frozen=True)
class SamplerConfig:
    negative_fraction: float = 0.45
    mined_negative_share: float = 0.45
    boundary_negative_share: float = 0.35
    near_negative_share: float = 0.15
    far_negative_share: float = 0.05
    boundary_radius: int = 8
    hard_negative_min_score: float = 0.20
    hard_negative_top_k_per_case: int = 32
    unilateral_positive_boost: float = 1.50
    seed: int = 42

    def validate(self) -> None:
        if not 0 <= self.negative_fraction < 1:
            raise ValueError("negative_fraction must be in [0,1)")
        if self.boundary_radius < 0:
            raise ValueError("boundary_radius must be non-negative")
        if self.unilateral_positive_boost <= 0:
            raise ValueError("unilateral_positive_boost must be positive")
        shares = [
            self.mined_negative_share,
            self.boundary_negative_share,
            self.near_negative_share,
            self.far_negative_share,
        ]
        if any(v < 0 for v in shares) or (
            self.negative_fraction > 0 and sum(shares) <= 0
        ):
            raise ValueError("invalid negative category shares")


def _frame(rows: Iterable[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows)).copy()
    required = {
        "case_id", "slice_idx",
        "left_presence", "right_presence",
        "left_area_bin", "right_area_bin",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Manifest missing columns: {missing}")
    frame["case_id"] = frame["case_id"].astype(str)
    frame["slice_idx"] = frame["slice_idx"].astype(int)
    for side in SIDES:
        frame[f"{side}_presence"] = frame[f"{side}_presence"].astype(int)
    frame["has_any"] = (
        (frame["left_presence"] == 1)
        | (frame["right_presence"] == 1)
    ).astype(int)
    return frame.reset_index(drop=True)


def _nearest_any_positive_distance(frame: pd.DataFrame) -> np.ndarray:
    output = np.full(len(frame), np.inf, dtype=np.float64)
    for _, indices in frame.groupby("case_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        case = frame.loc[idx].sort_values("slice_idx")
        ordered_idx = case.index.to_numpy(dtype=int)
        slices = case["slice_idx"].to_numpy(dtype=int)
        positive = slices[case["has_any"].to_numpy(dtype=bool)]
        if len(positive):
            output[ordered_idx] = np.min(
                np.abs(slices[:, None] - positive[None, :]),
                axis=1,
            )
    return output


def _probability_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for left, right in (
        ("left_smooth_presence_probability", "right_smooth_presence_probability"),
        ("left_raw_presence_probability", "right_raw_presence_probability"),
        ("left_presence_probability", "right_presence_probability"),
    ):
        if left in frame.columns and right in frame.columns:
            return left, right
    raise KeyError("Hard-negative CSV has no bilateral presence probabilities")


def load_hard_negative_keys(
    path: str | Path | None,
    *,
    minimum_score: float,
    top_k_per_case: int,
) -> tuple[set[tuple[str, int]], dict]:
    if path is None:
        return set(), {"source": None, "selected_rows": 0}

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    frame["case_id"] = frame["case_id"].astype(str)
    frame["slice_idx"] = frame["slice_idx"].astype(int)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == "train"].copy()

    left_col, right_col = _probability_columns(frame)
    frame["score"] = frame[[left_col, right_col]].max(axis=1)

    if {"left_gt_presence", "right_gt_presence"}.issubset(frame.columns):
        frame = frame[
            (frame["left_gt_presence"].astype(int) == 0)
            & (frame["right_gt_presence"].astype(int) == 0)
        ]
    elif {"left_presence", "right_presence"}.issubset(frame.columns):
        frame = frame[
            (frame["left_presence"].astype(int) == 0)
            & (frame["right_presence"].astype(int) == 0)
        ]

    eligible = frame[frame["score"] >= float(minimum_score)].copy()
    eligible = eligible.sort_values(
        ["case_id", "score", "slice_idx"],
        ascending=[True, False, True],
    )
    if top_k_per_case > 0:
        selected = eligible.groupby("case_id", sort=False).head(top_k_per_case)
    else:
        selected = eligible

    keys = set(zip(selected["case_id"], selected["slice_idx"]))
    return keys, {
        "source": str(source),
        "probability_columns": [left_col, right_col],
        "minimum_score": float(minimum_score),
        "top_k_per_case": int(top_k_per_case),
        "eligible_rows": int(len(eligible)),
        "selected_rows": int(len(selected)),
        "selected_cases": int(selected["case_id"].nunique()) if len(selected) else 0,
    }


def classify_rows(
    rows: Iterable[dict[str, str]],
    *,
    boundary_radius: int,
    hard_negative_keys: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    frame = _frame(rows)
    hard_negative_keys = hard_negative_keys or set()
    frame["nearest_positive_distance"] = _nearest_any_positive_distance(frame)
    if "is_near_negative" in frame.columns:
        near = frame["is_near_negative"].astype(int).eq(1)
    else:
        near = pd.Series(False, index=frame.index)

    categories = []
    for index, row in frame.iterrows():
        left = int(row["left_presence"])
        right = int(row["right_presence"])
        if left or right:
            categories.append(
                "positive_bilateral" if left and right else "positive_unilateral"
            )
            continue
        key = (str(row["case_id"]), int(row["slice_idx"]))
        if key in hard_negative_keys:
            categories.append("negative_mined")
        elif float(row["nearest_positive_distance"]) <= boundary_radius:
            categories.append("negative_boundary")
        elif bool(near.iloc[index]):
            categories.append("negative_near")
        else:
            categories.append("negative_far")

    frame["sampling_category"] = categories
    return frame


def _assign_mass(
    weights: np.ndarray,
    indices: np.ndarray,
    target_mass: float,
) -> None:
    if len(indices) == 0 or target_mass <= 0:
        return
    total = float(weights[indices].sum())
    if total <= 0:
        weights[indices] = target_mass / len(indices)
    else:
        weights[indices] *= target_mass / total


def build_presence_repair_sampler(
    dataset,
    *,
    config: SamplerConfig,
    hard_negative_csv: str | Path | None = None,
) -> tuple[WeightedRandomSampler, dict, pd.DataFrame]:
    config.validate()
    hard_keys, hard_audit = load_hard_negative_keys(
        hard_negative_csv,
        minimum_score=config.hard_negative_min_score,
        top_k_per_case=config.hard_negative_top_k_per_case,
    )
    frame = classify_rows(
        dataset.rows,
        boundary_radius=config.boundary_radius,
        hard_negative_keys=hard_keys,
    )

    positive_idx = np.flatnonzero(frame["has_any"].to_numpy() == 1)
    negative_idx = np.flatnonzero(frame["has_any"].to_numpy() == 0)
    if len(positive_idx) == 0:
        raise RuntimeError("No positive training rows")
    if config.negative_fraction > 0 and len(negative_idx) == 0:
        raise RuntimeError("No negative training rows")

    weights = np.zeros(len(frame), dtype=np.float64)
    side_bin_counts: Counter[tuple[str, str]] = Counter()
    for index in positive_idx:
        row = frame.iloc[index]
        for side in SIDES:
            if int(row[f"{side}_presence"]):
                side_bin_counts[(side, str(row[f"{side}_area_bin"]))] += 1

    for index in positive_idx:
        row = frame.iloc[index]
        contributions = []
        for side in SIDES:
            if int(row[f"{side}_presence"]):
                key = (side, str(row[f"{side}_area_bin"]))
                contributions.append(1.0 / max(side_bin_counts[key], 1))
        weight = float(np.mean(contributions)) if contributions else 1.0
        if row["sampling_category"] == "positive_unilateral":
            weight *= config.unilateral_positive_boost
        weights[index] = weight

    positive_mass = 1.0 - config.negative_fraction
    _assign_mass(weights, positive_idx, positive_mass)

    requested = {
        "negative_mined": config.mined_negative_share,
        "negative_boundary": config.boundary_negative_share,
        "negative_near": config.near_negative_share,
        "negative_far": config.far_negative_share,
    }
    groups = {
        category: np.flatnonzero(frame["sampling_category"].to_numpy() == category)
        for category in NEGATIVE_CATEGORIES
    }
    denominator = sum(
        requested[category]
        for category, indices in groups.items()
        if len(indices)
    )
    if config.negative_fraction > 0 and denominator <= 0:
        raise RuntimeError("No negative category can receive sampling mass")

    target_masses = {}
    for category, indices in groups.items():
        mass = (
            config.negative_fraction * requested[category] / denominator
            if len(indices) and config.negative_fraction > 0
            else 0.0
        )
        target_masses[category] = float(mass)
        if len(indices):
            weights[indices] = 1.0
            _assign_mass(weights, indices, mass)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("Invalid sampler weights")

    num_samples = int(round(len(positive_idx) / max(positive_mass, 1e-8)))
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=max(num_samples, 1),
        replacement=True,
        generator=generator,
    )

    frame["sampling_weight"] = weights
    frame["sampling_probability"] = weights / weights.sum()
    audit = {
        "config": asdict(config),
        "rows": int(len(frame)),
        "positive_rows": int(len(positive_idx)),
        "negative_rows": int(len(negative_idx)),
        "num_samples_per_epoch": int(num_samples),
        "category_counts": {
            str(k): int(v)
            for k, v in frame["sampling_category"].value_counts().to_dict().items()
        },
        "category_probability_mass": {
            str(k): float(v)
            for k, v in frame.groupby("sampling_category")[
                "sampling_probability"
            ].sum().to_dict().items()
        },
        "negative_target_mass": target_masses,
        "hard_negative_source": hard_audit,
    }
    return sampler, audit, frame
