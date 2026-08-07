from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a3b"))

from v1a4.recovery_decoder import (
    decode_hysteresis_recovery,
    decode_satellite_recovery,
    frozen_b1_seed,
)


def main() -> None:
    # Hysteresis should extend a valid dominant seed into adjacent lower-confidence
    # boundary evidence without creating a disconnected island.
    p = np.asarray(
        [0.0, 0.0010, 0.0015, 0.0040, 0.0080, 0.0100, 0.0020, 0.0015, 0.0],
        dtype=float,
    )
    seed = frozen_b1_seed(p, side="left")
    recovered = decode_hysteresis_recovery(
        p,
        side="left",
        recovery_threshold=0.0010,
        max_extension=3,
        bridge_gap=0,
    )
    assert recovered.sum() >= seed.sum()
    assert recovered[1] == 1
    assert recovered[-1] == 0

    # Satellite recovery should retain one nearby secondary segment but reject a
    # distant high-confidence island.
    q = np.asarray(
        [
            0.0,
            0.010, 0.012, 0.011,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.006, 0.007,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.020, 0.020,
            0.0,
        ],
        dtype=float,
    )
    sat = decode_satellite_recovery(
        q,
        side="left",
        satellite_max_distance=6,
        satellite_evidence_ratio=0.05,
        satellite_min_length=1,
        max_satellites=1,
    )
    assert sat[1:4].any()
    assert sat[9:11].any()
    assert not sat[18:20].any()

    print("V1a.4 recovery-decoder synthetic gate: PASSED")


if __name__ == "__main__":
    main()
