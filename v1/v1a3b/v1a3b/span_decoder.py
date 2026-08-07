from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

SIDES = ("left", "right")


@dataclass(frozen=True)
class SpanDecoderConfig:
    mode: str
    left_threshold: float
    right_threshold: float
    baseline_max_gap: int = 2
    max_internal_gap: int = 2
    min_span_length: int = 1
    span_penalty: float = 0.0
    length_penalty: float = 0.0

    def threshold_for(self, side: str) -> float:
        return self.left_threshold if side == "left" else self.right_threshold

    def to_dict(self) -> dict:
        return asdict(self)


def contiguous_segments(mask: Iterable[int | bool]) -> list[tuple[int, int]]:
    values = np.asarray(list(mask), dtype=bool)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(values) - 1))
    return segments


def _remove_isolated_positives(present: np.ndarray) -> np.ndarray:
    output = present.astype(bool, copy=True)
    if len(output) < 3:
        return output
    for i in range(1, len(output) - 1):
        if output[i] and not output[i - 1] and not output[i + 1]:
            output[i] = False
    return output


def _fill_short_gaps(present: np.ndarray, max_gap: int) -> np.ndarray:
    output = present.astype(bool, copy=True)
    i = 0
    while i < len(output):
        if output[i]:
            i += 1
            continue
        start = i
        while i < len(output) and not output[i]:
            i += 1
        gap = i - start
        if start > 0 and i < len(output) and gap <= int(max_gap):
            output[start:i] = True
    return output


def decode_baseline(probability: np.ndarray, threshold: float, max_gap: int) -> np.ndarray:
    present = np.asarray(probability, dtype=float) >= float(threshold)
    present = _remove_isolated_positives(present)
    present = _fill_short_gaps(present, max_gap=max_gap)
    return present.astype(np.uint8)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values) - np.log1p(-values)


def decode_single_span(
    probability: np.ndarray,
    threshold: float,
    max_gap: int,
    min_span_length: int = 1,
) -> np.ndarray:
    """Keep the single strongest segment after baseline threshold/gap processing.

    Strength is the summed positive log-odds evidence relative to the threshold.
    This decoder cannot create a new span; it only removes extra islands.
    """
    probability = np.asarray(probability, dtype=float)
    baseline = decode_baseline(probability, threshold=threshold, max_gap=max_gap)
    segments = [
        (start, end)
        for start, end in contiguous_segments(baseline)
        if end - start + 1 >= int(min_span_length)
    ]
    output = np.zeros_like(baseline, dtype=np.uint8)
    if not segments:
        return output

    evidence = _logit(probability) - float(_logit(np.asarray([threshold]))[0])
    best = max(
        segments,
        key=lambda item: (
            float(np.maximum(evidence[item[0] : item[1] + 1], 0.0).sum()),
            item[1] - item[0] + 1,
            float(probability[item[0] : item[1] + 1].max()),
        ),
    )
    output[best[0] : best[1] + 1] = 1
    return output


def _allowed_blocks(high: np.ndarray, max_internal_gap: int) -> list[tuple[int, int]]:
    """Split sequence at low-confidence runs longer than max_internal_gap."""
    n = len(high)
    if n == 0:
        return []
    max_gap = int(max_internal_gap)
    if max_gap < 0:
        raise ValueError("max_internal_gap must be non-negative")

    blocks: list[tuple[int, int]] = []
    block_start = 0
    i = 0
    while i < n:
        if high[i]:
            i += 1
            continue
        run_start = i
        while i < n and not high[i]:
            i += 1
        run_end = i - 1
        run_length = run_end - run_start + 1
        if run_length > max_gap:
            left_end = run_start - 1
            if left_end >= block_start:
                blocks.append((block_start, left_end))
            block_start = run_end + 1
    if block_start < n:
        blocks.append((block_start, n - 1))
    return blocks


def decode_optimal_span(
    probability: np.ndarray,
    threshold: float,
    max_internal_gap: int = 2,
    min_span_length: int = 1,
    span_penalty: float = 0.0,
    length_penalty: float = 0.0,
) -> np.ndarray:
    """Choose one contiguous anatomical span using threshold-relative log evidence.

    Candidate spans must begin and end at slices at or above `threshold`, and may
    cross at most `max_internal_gap` consecutive below-threshold slices. Long low
    runs split the sequence into independent blocks. Empty output is allowed when
    the best penalized score is not positive.
    """
    probability = np.asarray(probability, dtype=float)
    n = len(probability)
    output = np.zeros(n, dtype=np.uint8)
    if n == 0:
        return output
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("optimal-span threshold must be strictly between 0 and 1")

    high = probability >= threshold
    if not high.any():
        return output

    evidence = _logit(probability) - float(_logit(np.asarray([threshold]))[0])
    evidence = evidence - float(length_penalty)
    prefix = np.concatenate([[0.0], np.cumsum(evidence)])

    best_score = 0.0
    best_span: tuple[int, int] | None = None
    min_length = max(int(min_span_length), 1)

    for block_start, block_end in _allowed_blocks(high, max_internal_gap=max_internal_gap):
        high_indices = np.flatnonzero(high[block_start : block_end + 1]) + block_start
        if not len(high_indices):
            continue
        for start_pos, start in enumerate(high_indices):
            ends = high_indices[start_pos:]
            for end in ends:
                length = int(end - start + 1)
                if length < min_length:
                    continue
                score = float(prefix[end + 1] - prefix[start] - float(span_penalty))
                if score > best_score:
                    best_score = score
                    best_span = (int(start), int(end))
                elif score == best_score and best_span is not None:
                    old_length = best_span[1] - best_span[0] + 1
                    if length < old_length:
                        best_span = (int(start), int(end))

    if best_span is not None:
        output[best_span[0] : best_span[1] + 1] = 1
    return output


def decode_side(probability: np.ndarray, config: SpanDecoderConfig, side: str) -> np.ndarray:
    threshold = config.threshold_for(side)
    if config.mode == "baseline":
        return decode_baseline(probability, threshold=threshold, max_gap=config.baseline_max_gap)
    if config.mode == "single_span":
        return decode_single_span(
            probability,
            threshold=threshold,
            max_gap=config.baseline_max_gap,
            min_span_length=config.min_span_length,
        )
    if config.mode == "optimal_span":
        return decode_optimal_span(
            probability,
            threshold=threshold,
            max_internal_gap=config.max_internal_gap,
            min_span_length=config.min_span_length,
            span_penalty=config.span_penalty,
            length_penalty=config.length_penalty,
        )
    raise ValueError(f"Unknown decoder mode: {config.mode}")


def apply_span_decoder(frame: pd.DataFrame, config: SpanDecoderConfig) -> pd.DataFrame:
    required = {
        "case_id",
        "slice_idx",
        "left_smooth_presence_probability",
        "right_smooth_presence_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing decoder columns: {missing}")

    output = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    for side in SIDES:
        output[f"{side}_smooth_presence"] = 0
        output[f"{side}_span_presence"] = 0

    for _, indices in output.groupby("case_id", sort=False).groups.items():
        ordered_indices = np.asarray(list(indices), dtype=int)
        case = output.loc[ordered_indices].sort_values("slice_idx")
        ordered_indices = case.index.to_numpy(dtype=int)
        for side in SIDES:
            probability = case[f"{side}_smooth_presence_probability"].to_numpy(dtype=float)
            present = decode_side(probability, config=config, side=side)
            output.loc[ordered_indices, f"{side}_smooth_presence"] = present
            output.loc[ordered_indices, f"{side}_span_presence"] = present

    for side in SIDES:
        output[f"{side}_smooth_presence"] = output[f"{side}_smooth_presence"].astype(np.uint8)
        output[f"{side}_span_presence"] = output[f"{side}_span_presence"].astype(np.uint8)
    return output


def _safe_mean(values: list[float]) -> float:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return float(finite.mean()) if len(finite) else float("nan")


def summarize_presence_topology(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required = {
        "case_id",
        "slice_idx",
        "left_gt_presence",
        "right_gt_presence",
        "left_smooth_presence",
        "right_smooth_presence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing topology columns: {missing}")

    case_side_rows: list[dict] = []
    false_segment_rows: list[dict] = []
    tp = fp = fn = tn = 0

    ordered_frame = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True)
    for case_id, case in ordered_frame.groupby("case_id", sort=False):
        case = case.sort_values("slice_idx")
        slice_indices = case["slice_idx"].to_numpy(dtype=int)
        for side in SIDES:
            gt = case[f"{side}_gt_presence"].to_numpy(dtype=int).astype(bool)
            pred = case[f"{side}_smooth_presence"].to_numpy(dtype=int).astype(bool)
            tp += int(np.logical_and(gt, pred).sum())
            fp += int(np.logical_and(~gt, pred).sum())
            fn += int(np.logical_and(gt, ~pred).sum())
            tn += int(np.logical_and(~gt, ~pred).sum())

            gt_segments = contiguous_segments(gt)
            pred_segments = contiguous_segments(pred)
            gt_start = int(slice_indices[np.flatnonzero(gt)[0]]) if gt.any() else None
            gt_end = int(slice_indices[np.flatnonzero(gt)[-1]]) if gt.any() else None
            pred_start = int(slice_indices[np.flatnonzero(pred)[0]]) if pred.any() else None
            pred_end = int(slice_indices[np.flatnonzero(pred)[-1]]) if pred.any() else None

            fp_segments = 0
            fp_segment_slices = 0
            for segment_index, (start, end) in enumerate(pred_segments):
                segment_gt = gt[start : end + 1]
                if segment_gt.any():
                    continue
                fp_segments += 1
                length = end - start + 1
                fp_segment_slices += length
                z_start = int(slice_indices[start])
                z_end = int(slice_indices[end])
                if gt_start is None:
                    relation = "no_gt_span"
                    distance = None
                elif z_end < gt_start:
                    relation = "before_gt"
                    distance = int(gt_start - z_end)
                elif z_start > gt_end:
                    relation = "after_gt"
                    distance = int(z_start - gt_end)
                else:
                    relation = "inside_gt_extent"
                    distance = 0
                false_segment_rows.append(
                    {
                        "case_id": str(case_id),
                        "side": side,
                        "segment_index": int(segment_index),
                        "z_start": z_start,
                        "z_end": z_end,
                        "length": int(length),
                        "relation": relation,
                        "distance_to_gt_span": distance,
                    }
                )

            case_side_rows.append(
                {
                    "case_id": str(case_id),
                    "side": side,
                    "num_slices": int(len(case)),
                    "gt_positive_slices": int(gt.sum()),
                    "pred_positive_slices": int(pred.sum()),
                    "gt_segment_count": int(len(gt_segments)),
                    "pred_segment_count": int(len(pred_segments)),
                    "pred_multi_segment": int(len(pred_segments) > 1),
                    "false_positive_segment_count": int(fp_segments),
                    "false_positive_segment_slices": int(fp_segment_slices),
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "onset_error": float(pred_start - gt_start) if gt_start is not None and pred_start is not None else np.nan,
                    "offset_error": float(pred_end - gt_end) if gt_end is not None and pred_end is not None else np.nan,
                    "onset_abs_error": float(abs(pred_start - gt_start)) if gt_start is not None and pred_start is not None else np.nan,
                    "offset_abs_error": float(abs(pred_end - gt_end)) if gt_end is not None and pred_end is not None else np.nan,
                }
            )

    case_side = pd.DataFrame(case_side_rows)
    false_segments = pd.DataFrame(false_segment_rows)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    summary = {
        "side_samples": int(tp + fp + fn + tn),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "positive_recall": float(recall),
        "false_positive_rate": float(fpr),
        "specificity": float(1.0 - fpr),
        "case_side_count": int(len(case_side)),
        "multi_segment_case_side_rate": float(case_side["pred_multi_segment"].mean()) if len(case_side) else float("nan"),
        "case_sides_with_false_positive_segments": int((case_side["false_positive_segment_count"] > 0).sum()) if len(case_side) else 0,
        "false_positive_segment_count": int(case_side["false_positive_segment_count"].sum()) if len(case_side) else 0,
        "false_positive_segment_slices": int(case_side["false_positive_segment_slices"].sum()) if len(case_side) else 0,
        "onset_abs_error_mean": _safe_mean(case_side["onset_abs_error"].tolist()) if len(case_side) else float("nan"),
        "offset_abs_error_mean": _safe_mean(case_side["offset_abs_error"].tolist()) if len(case_side) else float("nan"),
        "onset_abs_error_median": float(case_side["onset_abs_error"].median()) if len(case_side) else float("nan"),
        "offset_abs_error_median": float(case_side["offset_abs_error"].median()) if len(case_side) else float("nan"),
    }
    for side in SIDES:
        side_rows = case_side[case_side["side"] == side]
        detail_mask = ordered_frame[f"{side}_gt_presence"].to_numpy(dtype=int)
        pred_mask = ordered_frame[f"{side}_smooth_presence"].to_numpy(dtype=int)
        side_tp = int(np.logical_and(detail_mask == 1, pred_mask == 1).sum())
        side_fn = int(np.logical_and(detail_mask == 1, pred_mask == 0).sum())
        side_fp = int(np.logical_and(detail_mask == 0, pred_mask == 1).sum())
        side_tn = int(np.logical_and(detail_mask == 0, pred_mask == 0).sum())
        summary[f"{side}_positive_recall"] = float(side_tp / max(side_tp + side_fn, 1))
        summary[f"{side}_false_positive_rate"] = float(side_fp / max(side_fp + side_tn, 1))
        summary[f"{side}_multi_segment_case_side_rate"] = float(side_rows["pred_multi_segment"].mean()) if len(side_rows) else float("nan")
    return summary, case_side, false_segments
