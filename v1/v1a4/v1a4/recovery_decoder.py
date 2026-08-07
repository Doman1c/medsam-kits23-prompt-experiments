from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from v1a3b.span_decoder import (
    SIDES,
    contiguous_segments,
    decode_baseline,
    decode_single_span,
)

FROZEN_THRESHOLDS = {
    "left": 0.0035,
    "right": 0.0400,
}
FROZEN_BASELINE_MAX_GAP = 4
FROZEN_BOUNDARY_PAD = 1


@dataclass(frozen=True)
class RecoveryDecoderConfig:
    mode: str
    left_recovery_threshold: float = 0.0
    right_recovery_threshold: float = 0.0
    max_extension: int = 0
    bridge_gap: int = 0
    satellite_max_distance: int = 0
    satellite_evidence_ratio: float = 0.0
    satellite_min_length: int = 1
    max_satellites: int = 1

    def recovery_threshold_for(self, side: str) -> float:
        if side == "left":
            return float(self.left_recovery_threshold)
        if side == "right":
            return float(self.right_recovery_threshold)
        raise ValueError(side)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values) - np.log1p(-values)


def _pad_each_segment(mask: np.ndarray, pad: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be non-negative")
    if pad == 0 or not mask.any():
        return mask.copy()

    output = np.zeros_like(mask, dtype=np.uint8)
    for start, end in contiguous_segments(mask):
        left = max(int(start) - pad, 0)
        right = min(int(end) + pad, len(mask) - 1)
        output[left : right + 1] = 1
    return output


def frozen_b1_seed(probability: np.ndarray, side: str) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    threshold = float(FROZEN_THRESHOLDS[side])
    seed = decode_single_span(
        probability,
        threshold=threshold,
        max_gap=FROZEN_BASELINE_MAX_GAP,
        min_span_length=1,
    )
    return _pad_each_segment(seed, FROZEN_BOUNDARY_PAD)


def _extend_one_boundary(
    probability: np.ndarray,
    output: np.ndarray,
    boundary: int,
    direction: int,
    recovery_threshold: float,
    max_extension: int,
    bridge_gap: int,
) -> None:
    n = len(probability)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    max_extension = int(max_extension)
    bridge_gap = int(bridge_gap)
    if max_extension < 0 or bridge_gap < 0:
        raise ValueError("max_extension and bridge_gap must be non-negative")

    farthest_accepted: int | None = None
    below_run = 0
    pos = int(boundary) + direction
    traversed = 0

    while 0 <= pos < n and traversed < max_extension:
        traversed += 1
        if float(probability[pos]) >= float(recovery_threshold):
            farthest_accepted = int(pos)
            below_run = 0
        else:
            below_run += 1
            if below_run > bridge_gap:
                break
        pos += direction

    if farthest_accepted is None:
        return

    if direction < 0:
        output[farthest_accepted : boundary + 1] = 1
    else:
        output[boundary : farthest_accepted + 1] = 1


def decode_hysteresis_recovery(
    probability: np.ndarray,
    side: str,
    recovery_threshold: float,
    max_extension: int,
    bridge_gap: int,
) -> np.ndarray:
    """Recover only slices adjacent to the frozen B1 dominant span.

    The frozen B1 span acts as the high-confidence seed. Each boundary may extend
    outward at a lower probability threshold, with a bounded number of consecutive
    below-threshold bridge slices. No new disconnected segment can be created.
    """
    probability = np.asarray(probability, dtype=float)
    output = frozen_b1_seed(probability, side=side)
    positive = np.flatnonzero(output)
    if not len(positive):
        return output

    _extend_one_boundary(
        probability,
        output,
        boundary=int(positive[0]),
        direction=-1,
        recovery_threshold=float(recovery_threshold),
        max_extension=int(max_extension),
        bridge_gap=int(bridge_gap),
    )
    positive = np.flatnonzero(output)
    _extend_one_boundary(
        probability,
        output,
        boundary=int(positive[-1]),
        direction=1,
        recovery_threshold=float(recovery_threshold),
        max_extension=int(max_extension),
        bridge_gap=int(bridge_gap),
    )
    return output.astype(np.uint8)


def _segment_evidence(
    probability: np.ndarray,
    segment: tuple[int, int],
    threshold: float,
) -> float:
    start, end = segment
    relative = _safe_logit(probability[start : end + 1]) - float(
        _safe_logit(np.asarray([threshold]))[0]
    )
    return float(np.maximum(relative, 0.0).sum())


def _segment_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] < b[0]:
        return int(b[0] - a[1] - 1)
    if b[1] < a[0]:
        return int(a[0] - b[1] - 1)
    return 0


def decode_satellite_recovery(
    probability: np.ndarray,
    side: str,
    satellite_max_distance: int,
    satellite_evidence_ratio: float,
    satellite_min_length: int,
    max_satellites: int = 1,
) -> np.ndarray:
    """Retain the frozen dominant span plus a limited nearby baseline segment.

    Satellites are proposed only from the frozen-threshold baseline mask. A segment
    is eligible when it is close to the dominant span, long enough, and carries a
    minimum fraction of the dominant segment's threshold-relative evidence. This
    can recover a split anatomical span without reintroducing distant FP islands.
    """
    probability = np.asarray(probability, dtype=float)
    threshold = float(FROZEN_THRESHOLDS[side])

    dominant_mask = decode_single_span(
        probability,
        threshold=threshold,
        max_gap=FROZEN_BASELINE_MAX_GAP,
        min_span_length=1,
    )
    dominant_segments = contiguous_segments(dominant_mask)
    if not dominant_segments:
        return np.zeros(len(probability), dtype=np.uint8)
    dominant = dominant_segments[0]

    baseline = decode_baseline(
        probability,
        threshold=threshold,
        max_gap=FROZEN_BASELINE_MAX_GAP,
    )
    segments = contiguous_segments(baseline)
    dominant_evidence = max(
        _segment_evidence(probability, dominant, threshold),
        1e-12,
    )

    eligible: list[tuple[float, int, int, tuple[int, int]]] = []
    for segment in segments:
        if segment == dominant:
            continue
        length = int(segment[1] - segment[0] + 1)
        if length < int(satellite_min_length):
            continue
        distance = _segment_distance(segment, dominant)
        if distance > int(satellite_max_distance):
            continue
        evidence = _segment_evidence(probability, segment, threshold)
        ratio = float(evidence / dominant_evidence)
        if ratio < float(satellite_evidence_ratio):
            continue
        eligible.append((ratio, -distance, length, segment))

    eligible.sort(reverse=True)
    output = np.zeros(len(probability), dtype=np.uint8)
    output[dominant[0] : dominant[1] + 1] = 1
    for _, _, _, segment in eligible[: max(int(max_satellites), 0)]:
        output[segment[0] : segment[1] + 1] = 1

    return _pad_each_segment(output, FROZEN_BOUNDARY_PAD)


def decode_side(
    probability: np.ndarray,
    side: str,
    config: RecoveryDecoderConfig,
) -> np.ndarray:
    if config.mode == "frozen_b1":
        return frozen_b1_seed(probability, side=side)
    if config.mode == "hysteresis":
        return decode_hysteresis_recovery(
            probability,
            side=side,
            recovery_threshold=config.recovery_threshold_for(side),
            max_extension=config.max_extension,
            bridge_gap=config.bridge_gap,
        )
    if config.mode == "satellite":
        return decode_satellite_recovery(
            probability,
            side=side,
            satellite_max_distance=config.satellite_max_distance,
            satellite_evidence_ratio=config.satellite_evidence_ratio,
            satellite_min_length=config.satellite_min_length,
            max_satellites=config.max_satellites,
        )
    raise ValueError(f"Unknown recovery mode: {config.mode}")


def apply_recovery_decoder(
    frame: pd.DataFrame,
    config: RecoveryDecoderConfig,
) -> pd.DataFrame:
    required = {
        "case_id",
        "slice_idx",
        "left_smooth_presence_probability",
        "right_smooth_presence_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing recovery columns: {missing}")

    output = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    for side in SIDES:
        output[f"{side}_smooth_presence"] = 0
        output[f"{side}_span_presence"] = 0

    for _, indices in output.groupby("case_id", sort=False).groups.items():
        ordered = output.loc[list(indices)].sort_values("slice_idx")
        ordered_indices = ordered.index.to_numpy(dtype=int)
        for side in SIDES:
            probability = ordered[
                f"{side}_smooth_presence_probability"
            ].to_numpy(dtype=float)
            present = decode_side(probability, side=side, config=config)
            output.loc[ordered_indices, f"{side}_smooth_presence"] = present
            output.loc[ordered_indices, f"{side}_span_presence"] = present

    for side in SIDES:
        output[f"{side}_smooth_presence"] = output[
            f"{side}_smooth_presence"
        ].astype(np.uint8)
        output[f"{side}_span_presence"] = output[
            f"{side}_span_presence"
        ].astype(np.uint8)
    return output
