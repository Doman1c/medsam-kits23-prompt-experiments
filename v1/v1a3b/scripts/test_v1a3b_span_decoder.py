from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a3b.span_decoder import (
    SpanDecoderConfig,
    contiguous_segments,
    decode_baseline,
    decode_optimal_span,
    decode_single_span,
)


def main() -> None:
    probability = np.asarray([0.001, 0.02, 0.03, 0.001, 0.002, 0.025, 0.001], dtype=float)
    baseline = decode_baseline(probability, threshold=0.01, max_gap=0)
    assert contiguous_segments(baseline) == [(1, 2)]

    probability = np.asarray([0.02, 0.03, 0.001, 0.02, 0.03, 0.001, 0.001], dtype=float)
    single = decode_single_span(probability, threshold=0.01, max_gap=0)
    assert len(contiguous_segments(single)) == 1

    probability = np.asarray([0.02, 0.03, 0.004, 0.02, 0.03], dtype=float)
    optimal = decode_optimal_span(
        probability,
        threshold=0.01,
        max_internal_gap=1,
        min_span_length=1,
        span_penalty=0.0,
    )
    assert contiguous_segments(optimal) == [(0, 4)]

    blocked = decode_optimal_span(
        np.asarray([0.03, 0.03, 0.001, 0.001, 0.03, 0.03]),
        threshold=0.01,
        max_internal_gap=1,
    )
    assert len(contiguous_segments(blocked)) == 1
    assert not (blocked[0] and blocked[-1]), "decoder crossed a forbidden two-slice low-confidence gap"

    config = SpanDecoderConfig(mode="optimal_span", left_threshold=0.01, right_threshold=0.01)
    assert config.mode == "optimal_span"
    print("V1a.3-B span-decoder synthetic gate: PASSED")


if __name__ == "__main__":
    main()
