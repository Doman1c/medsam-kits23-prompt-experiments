from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIDES = ("left", "right")
BOX_KEYS = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class GeometryRecoveryConfig:
    mode: str
    trend_anchor_count: int = 5


def _box_columns(side: str, variant: str = "smooth") -> list[str]:
    return [f"{side}_{variant}_{key}" for key in BOX_KEYS]


def _presence_column(side: str, variant: str = "smooth") -> str:
    return f"{side}_{variant}_presence"


def _box_to_params(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    return np.asarray(
        [
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            max(float(x2 - x1), 2.0),
            max(float(y2 - y1), 2.0),
        ],
        dtype=float,
    )


def _params_to_box(params: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, bw, bh = np.asarray(params, dtype=float)
    width = int(width)
    height = int(height)
    bw = float(np.clip(bw, 2.0, max(width, 2)))
    bh = float(np.clip(bh, 2.0, max(height, 2)))
    cx = float(np.clip(cx, 0.0, max(width - 1, 0)))
    cy = float(np.clip(cy, 0.0, max(height - 1, 0)))
    x1 = float(np.clip(cx - bw / 2.0, 0.0, max(width - 1, 0)))
    y1 = float(np.clip(cy - bh / 2.0, 0.0, max(height - 1, 0)))
    x2 = float(np.clip(cx + bw / 2.0, x1 + 1.0, max(width, 1)))
    y2 = float(np.clip(cy + bh / 2.0, y1 + 1.0, max(height, 1)))
    return np.asarray([x1, y1, x2, y2], dtype=float)


def _validate_pair(b1: pd.DataFrame, r2b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"case_id", "slice_idx", "image_width", "image_height"}
    for side in SIDES:
        required.add(_presence_column(side))
        required.update(_box_columns(side))
    for name, frame in (("b1", b1), ("r2b", r2b)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    b1 = b1.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    r2b = r2b.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    b1_keys = list(zip(b1["case_id"].astype(str), b1["slice_idx"].astype(int)))
    r2b_keys = list(zip(r2b["case_id"].astype(str), r2b["slice_idx"].astype(int)))
    if b1_keys != r2b_keys:
        raise RuntimeError("B1 and R2b rows do not match exactly")
    return b1, r2b


def recovered_side_mask(
    b1: pd.DataFrame,
    r2b: pd.DataFrame,
    side: str,
) -> np.ndarray:
    if side not in SIDES:
        raise ValueError(side)
    b1, r2b = _validate_pair(b1, r2b)
    b1_present = b1[_presence_column(side)].to_numpy(dtype=int) > 0
    r2b_present = r2b[_presence_column(side)].to_numpy(dtype=int) > 0
    return np.logical_and(r2b_present, ~b1_present)


def _nearest_anchor_box(
    ordered_b1: pd.DataFrame,
    side: str,
    target_slice: int,
) -> tuple[np.ndarray, int]:
    present = ordered_b1[_presence_column(side)].to_numpy(dtype=int) > 0
    anchors = ordered_b1.loc[present].copy()
    if anchors.empty:
        raise RuntimeError(f"No frozen-B1 anchor for side={side}")
    distances = np.abs(anchors["slice_idx"].to_numpy(dtype=int) - int(target_slice))
    nearest_pos = int(np.argmin(distances))
    row = anchors.iloc[nearest_pos]
    box = row[_box_columns(side)].to_numpy(dtype=float)
    return box, int(distances[nearest_pos])


def _trend_box(
    ordered_b1: pd.DataFrame,
    side: str,
    target_slice: int,
    anchor_count: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int]:
    present = ordered_b1[_presence_column(side)].to_numpy(dtype=int) > 0
    anchors = ordered_b1.loc[present].copy()
    if anchors.empty:
        raise RuntimeError(f"No frozen-B1 anchor for side={side}")

    anchor_count = max(int(anchor_count), 2)
    distances = np.abs(anchors["slice_idx"].to_numpy(dtype=int) - int(target_slice))
    order = np.argsort(distances)[: min(anchor_count, len(anchors))]
    selected = anchors.iloc[order].sort_values("slice_idx")
    nearest_distance = int(np.min(distances))

    if len(selected) < 2 or selected["slice_idx"].nunique() < 2:
        box, _ = _nearest_anchor_box(ordered_b1, side=side, target_slice=target_slice)
        return box, nearest_distance, int(len(selected))

    z = selected["slice_idx"].to_numpy(dtype=float)
    params = np.vstack(
        [_box_to_params(row[_box_columns(side)].to_numpy(dtype=float)) for _, row in selected.iterrows()]
    )
    predicted = []
    for column in range(params.shape[1]):
        slope, intercept = np.polyfit(z, params[:, column], deg=1)
        predicted.append(float(slope * float(target_slice) + intercept))
    return _params_to_box(np.asarray(predicted), width=width, height=height), nearest_distance, int(len(selected))


def materialize_geometry_candidate(
    b1: pd.DataFrame,
    r2b: pd.DataFrame,
    config: GeometryRecoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply geometry recovery only where R2b adds presence beyond frozen B1.

    Presence is copied exactly from R2b for all geometry-recovery candidates.
    Geometry on non-recovered side-slices is also copied exactly from R2b.
    """
    b1, r2b = _validate_pair(b1, r2b)
    output = r2b.copy()
    records: list[dict] = []

    if config.mode == "native":
        for side in SIDES:
            output[f"{side}_geometry_recovered"] = 0
            output[f"{side}_geometry_method"] = "native"
        return output, pd.DataFrame(records)

    if config.mode not in {"nearest_anchor", "linear_trend"}:
        raise ValueError(f"Unknown geometry recovery mode: {config.mode}")

    for side in SIDES:
        output[f"{side}_geometry_recovered"] = 0
        output[f"{side}_geometry_method"] = "native"

    for case_id, indices in output.groupby("case_id", sort=False).groups.items():
        idx = list(indices)
        ordered_b1 = b1.loc[idx].sort_values("slice_idx")
        ordered_r2b = output.loc[idx].sort_values("slice_idx")
        for row_index, row in ordered_r2b.iterrows():
            for side in SIDES:
                b1_present = int(b1.loc[row_index, _presence_column(side)]) > 0
                r2b_present = int(output.loc[row_index, _presence_column(side)]) > 0
                if not (r2b_present and not b1_present):
                    continue

                target_slice = int(row["slice_idx"])
                width = int(row["image_width"])
                height = int(row["image_height"])
                native_box = output.loc[row_index, _box_columns(side)].to_numpy(dtype=float)

                if config.mode == "nearest_anchor":
                    recovered_box, nearest_distance = _nearest_anchor_box(
                        ordered_b1,
                        side=side,
                        target_slice=target_slice,
                    )
                    anchor_count_used = 1
                else:
                    recovered_box, nearest_distance, anchor_count_used = _trend_box(
                        ordered_b1,
                        side=side,
                        target_slice=target_slice,
                        anchor_count=config.trend_anchor_count,
                        width=width,
                        height=height,
                    )

                output.loc[row_index, _box_columns(side)] = recovered_box
                output.loc[row_index, f"{side}_geometry_recovered"] = 1
                output.loc[row_index, f"{side}_geometry_method"] = config.mode
                records.append(
                    {
                        "case_id": str(case_id),
                        "slice_idx": target_slice,
                        "side": side,
                        "mode": config.mode,
                        "nearest_anchor_distance": int(nearest_distance),
                        "anchor_count_used": int(anchor_count_used),
                        "native_x1": float(native_box[0]),
                        "native_y1": float(native_box[1]),
                        "native_x2": float(native_box[2]),
                        "native_y2": float(native_box[3]),
                        "recovered_x1": float(recovered_box[0]),
                        "recovered_y1": float(recovered_box[1]),
                        "recovered_x2": float(recovered_box[2]),
                        "recovered_y2": float(recovered_box[3]),
                    }
                )

    audit = pd.DataFrame(records)
    return output, audit


def assert_nonrecovered_geometry_unchanged(
    original_r2b: pd.DataFrame,
    candidate: pd.DataFrame,
    b1: pd.DataFrame,
) -> None:
    b1, original_r2b = _validate_pair(b1, original_r2b)
    _, candidate = _validate_pair(b1, candidate)
    for side in SIDES:
        recovered = recovered_side_mask(b1, original_r2b, side=side)
        keep = ~recovered
        original = original_r2b.loc[keep, _box_columns(side)].to_numpy(dtype=float)
        changed = candidate.loc[keep, _box_columns(side)].to_numpy(dtype=float)
        if not np.allclose(original, changed, rtol=0.0, atol=1e-8, equal_nan=True):
            raise RuntimeError(f"Non-recovered {side} geometry changed")
        if not np.array_equal(
            original_r2b[_presence_column(side)].to_numpy(dtype=int),
            candidate[_presence_column(side)].to_numpy(dtype=int),
        ):
            raise RuntimeError(f"{side} presence changed during geometry recovery")
