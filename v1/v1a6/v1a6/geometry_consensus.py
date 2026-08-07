from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIDES = ("left", "right")
BOX_KEYS = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class ConsensusConfig:
    mode: str
    rolling_window: int = 5
    mad_threshold: float = 3.5
    minimum_inliers: int = 5


def box_to_geometry(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return np.asarray(
        [0.5 * (x1 + x2), 0.5 * (y1 + y2), width, height], dtype=float
    )


def clip_geometry_to_box(
    geometry: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    cx, cy, width, height = map(float, geometry)
    width = max(width, 1.0)
    height = max(height, 1.0)
    x1 = np.clip(cx - 0.5 * width, 0.0, max(image_width - 1.0, 0.0))
    y1 = np.clip(cy - 0.5 * height, 0.0, max(image_height - 1.0, 0.0))
    x2 = np.clip(cx + 0.5 * width, x1 + 1.0, image_width)
    y2 = np.clip(cy + 0.5 * height, y1 + 1.0, image_height)
    return np.asarray([x1, y1, x2, y2], dtype=float)


def contiguous_runs(slice_indices: np.ndarray) -> list[np.ndarray]:
    if len(slice_indices) == 0:
        return []
    order = np.argsort(slice_indices)
    sorted_values = np.asarray(slice_indices, dtype=int)[order]
    breaks = np.where(np.diff(sorted_values) != 1)[0] + 1
    chunks = np.split(order, breaks)
    return [chunk for chunk in chunks if len(chunk)]


def satellite_median_geometry(geometries: np.ndarray) -> np.ndarray:
    median = np.median(geometries, axis=0)
    return np.repeat(median[None, :], len(geometries), axis=0)


def rolling_median_geometry(geometries: np.ndarray, window: int = 5) -> np.ndarray:
    if window < 1 or window % 2 != 1:
        raise ValueError("rolling median window must be a positive odd integer")
    half = window // 2
    output = []
    for index in range(len(geometries)):
        start = max(0, index - half)
        stop = min(len(geometries), index + half + 1)
        output.append(np.median(geometries[start:stop], axis=0))
    return np.asarray(output, dtype=float)


def robust_linear_geometry(
    geometries: np.ndarray,
    slice_indices: np.ndarray,
    mad_threshold: float = 3.5,
    minimum_inliers: int = 5,
) -> np.ndarray:
    if len(geometries) == 1:
        return geometries.copy()

    transformed = geometries.astype(float).copy()
    transformed[:, 2:] = np.log(np.maximum(transformed[:, 2:], 1.0))
    median = np.median(transformed, axis=0)
    mad = np.median(np.abs(transformed - median[None, :]), axis=0)
    scale = 1.4826 * mad + 1e-6
    scores = np.max(np.abs((transformed - median[None, :]) / scale[None, :]), axis=1)
    inliers = np.where(scores <= float(mad_threshold))[0]

    required = min(max(int(minimum_inliers), 2), len(transformed))
    if len(inliers) < required:
        inliers = np.argsort(scores)[:required]

    z = np.asarray(slice_indices, dtype=float)
    z_center = float(z[inliers].mean())
    z_scale = max(float(np.std(z[inliers])), 1.0)
    fit_z = (z[inliers] - z_center) / z_scale
    eval_z = (z - z_center) / z_scale

    predicted = np.empty_like(transformed, dtype=float)
    for column in range(transformed.shape[1]):
        coefficient = np.polyfit(fit_z, transformed[inliers, column], deg=1)
        predicted[:, column] = np.polyval(coefficient, eval_z)

    predicted[:, 2:] = np.exp(predicted[:, 2:])
    return predicted


def apply_consensus(
    r2b_frame: pd.DataFrame,
    b1_frame: pd.DataFrame,
    config: ConsensusConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"case_id", "slice_idx", "image_width", "image_height"}
    for side in SIDES:
        required.add(f"{side}_smooth_presence")
        required.update(f"{side}_smooth_{key}" for key in BOX_KEYS)
    missing = sorted(required - set(r2b_frame.columns))
    if missing:
        raise ValueError(f"R2b predictions missing columns: {missing}")

    left = r2b_frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    right = b1_frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    left_keys = list(zip(left["case_id"].astype(str), left["slice_idx"].astype(int)))
    right_keys = list(zip(right["case_id"].astype(str), right["slice_idx"].astype(int)))
    if left_keys != right_keys:
        raise RuntimeError("R2b and frozen-B1 rows are not exactly aligned")

    recovered_rows: list[dict] = []
    for case_id, case_indices in left.groupby("case_id", sort=False).groups.items():
        case_order = left.loc[list(case_indices)].sort_values("slice_idx")
        ordered_indices = case_order.index.to_numpy(dtype=int)
        slice_values = case_order["slice_idx"].to_numpy(dtype=int)

        for side in SIDES:
            r2b_presence = case_order[f"{side}_smooth_presence"].to_numpy(dtype=np.uint8)
            b1_presence = right.loc[ordered_indices, f"{side}_smooth_presence"].to_numpy(dtype=np.uint8)
            recovered_local = np.where((r2b_presence == 1) & (b1_presence == 0))[0]
            if len(recovered_local) == 0:
                continue

            recovered_slice_values = slice_values[recovered_local]
            for segment_number, run_positions in enumerate(contiguous_runs(recovered_slice_values)):
                local_positions = recovered_local[run_positions]
                frame_indices = ordered_indices[local_positions]
                segment_slices = slice_values[local_positions]
                boxes = np.asarray(
                    [
                        [float(left.loc[index, f"{side}_smooth_{key}"]) for key in BOX_KEYS]
                        for index in frame_indices
                    ],
                    dtype=float,
                )
                geometries = np.asarray([box_to_geometry(box) for box in boxes], dtype=float)

                if config.mode == "native":
                    recovered_geometries = geometries
                elif config.mode == "satellite_median":
                    recovered_geometries = satellite_median_geometry(geometries)
                elif config.mode == "rolling_median":
                    recovered_geometries = rolling_median_geometry(
                        geometries, window=int(config.rolling_window)
                    )
                elif config.mode == "robust_linear":
                    recovered_geometries = robust_linear_geometry(
                        geometries,
                        segment_slices,
                        mad_threshold=float(config.mad_threshold),
                        minimum_inliers=int(config.minimum_inliers),
                    )
                else:
                    raise ValueError(f"Unknown consensus mode: {config.mode}")

                for frame_index, slice_idx, native_box, recovered_geometry in zip(
                    frame_indices,
                    segment_slices,
                    boxes,
                    recovered_geometries,
                ):
                    image_width = float(left.loc[frame_index, "image_width"])
                    image_height = float(left.loc[frame_index, "image_height"])
                    output_box = clip_geometry_to_box(
                        recovered_geometry,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if config.mode != "native":
                        for key, value in zip(BOX_KEYS, output_box):
                            left.loc[frame_index, f"{side}_smooth_{key}"] = float(value)

                    recovered_rows.append(
                        {
                            "case_id": str(case_id),
                            "slice_idx": int(slice_idx),
                            "side": side,
                            "segment_number": int(segment_number),
                            "segment_length": int(len(frame_indices)),
                            "gt_present": int(left.loc[frame_index, f"{side}_gt_presence"]),
                            "presence_probability": float(
                                left.loc[frame_index, f"{side}_smooth_presence_probability"]
                            ),
                            "native_x1": float(native_box[0]),
                            "native_y1": float(native_box[1]),
                            "native_x2": float(native_box[2]),
                            "native_y2": float(native_box[3]),
                            "output_x1": float(output_box[0]),
                            "output_y1": float(output_box[1]),
                            "output_x2": float(output_box[2]),
                            "output_y2": float(output_box[3]),
                        }
                    )

    recovered = pd.DataFrame(recovered_rows)
    return left, recovered


def assert_invariants(
    candidate: pd.DataFrame,
    r2b_frame: pd.DataFrame,
    b1_frame: pd.DataFrame,
) -> dict:
    candidate = candidate.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
    r2b = r2b_frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
    b1 = b1_frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
    if len(candidate) != len(r2b) or len(r2b) != len(b1):
        raise AssertionError("row count mismatch")

    changed_recovered = 0
    changed_non_recovered = 0
    for side in SIDES:
        if not np.array_equal(
            candidate[f"{side}_smooth_presence"].to_numpy(),
            r2b[f"{side}_smooth_presence"].to_numpy(),
        ):
            raise AssertionError(f"{side} presence changed")

        recovered = (
            (r2b[f"{side}_smooth_presence"].to_numpy(dtype=int) == 1)
            & (b1[f"{side}_smooth_presence"].to_numpy(dtype=int) == 0)
        )
        for key in BOX_KEYS:
            source = r2b[f"{side}_smooth_{key}"].to_numpy(dtype=float)
            output = candidate[f"{side}_smooth_{key}"].to_numpy(dtype=float)
            changed = ~np.isclose(source, output, rtol=0.0, atol=1e-9)
            changed_recovered += int((changed & recovered).sum())
            changed_non_recovered += int((changed & ~recovered).sum())

    if changed_non_recovered:
        raise AssertionError(f"non-recovered geometry changed: {changed_non_recovered}")
    return {
        "presence_identical_to_r2b": True,
        "changed_recovered_geometry_values": int(changed_recovered),
        "changed_non_recovered_geometry_values": int(changed_non_recovered),
    }
