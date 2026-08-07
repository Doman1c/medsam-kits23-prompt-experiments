from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a3b"))

from v1a4.stage1b_decoder import (
    Stage1bDecoderConfig,
    decode_stage1b_side,
)


def main() -> None:
    # Dominant segment near the end; a strong remote satellite should be retained
    # when the distance/evidence gate allows it.
    probability = np.asarray(
        [
            0.0,
            0.020, 0.030, 0.025, 0.020,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.090, 0.100, 0.090, 0.080,
            0.0,
        ],
        dtype=float,
    )
    remote = decode_stage1b_side(
        probability,
        side="left",
        config=Stage1bDecoderConfig(
            mode="remote_satellite",
            satellite_max_distance=32,
            satellite_evidence_ratio=0.05,
            satellite_min_length=2,
        ),
    )
    assert remote[1:5].any(), "remote supported segment should be recovered"
    assert remote[20:24].any(), "dominant segment should remain"

    hybrid = decode_stage1b_side(
        probability,
        side="left",
        config=Stage1bDecoderConfig(
            mode="hybrid",
            satellite_max_distance=32,
            satellite_evidence_ratio=0.05,
            satellite_min_length=2,
        ),
    )
    assert np.all(hybrid >= remote), "hybrid must contain remote-satellite output"

    print("V1a.4 Stage-1b decoder synthetic gate: PASSED")


if __name__ == "__main__":
    main()
