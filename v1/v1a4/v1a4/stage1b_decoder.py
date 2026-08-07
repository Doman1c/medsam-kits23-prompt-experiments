from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from v1a3b.span_decoder import SIDES
from v1a4.recovery_decoder import (
    decode_hysteresis_recovery,
    decode_satellite_recovery,
)

# Stage-1 selected local-recovery operating point: maximum validation recall
# among hysteresis configurations with FPR <= 0.15. It is frozen before the
# Stage-1b remote-satellite sweep.
FIXED_LOCAL_HYSTERESIS = {
    "left_recovery_threshold": 0.0005,
    "right_recovery_threshold": 0.0025,
    "max_extension": 6,
    "bridge_gap": 2,
}


@dataclass(frozen=True)
class Stage1bDecoderConfig:
    mode: str
    satellite_max_distance: int
    satellite_evidence_ratio: float
    satellite_min_length: int
    max_satellites: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_local_hysteresis(probability: np.ndarray, side: str) -> np.ndarray:
    threshold_key = f"{side}_recovery_threshold"
    return decode_hysteresis_recovery(
        probability,
        side=side,
        recovery_threshold=float(FIXED_LOCAL_HYSTERESIS[threshold_key]),
        max_extension=int(FIXED_LOCAL_HYSTERESIS["max_extension"]),
        bridge_gap=int(FIXED_LOCAL_HYSTERESIS["bridge_gap"]),
    )


def remote_satellite(probability: np.ndarray, side: str, config: Stage1bDecoderConfig) -> np.ndarray:
    return decode_satellite_recovery(
        probability,
        side=side,
        satellite_max_distance=int(config.satellite_max_distance),
        satellite_evidence_ratio=float(config.satellite_evidence_ratio),
        satellite_min_length=int(config.satellite_min_length),
        max_satellites=int(config.max_satellites),
    )


def decode_stage1b_side(
    probability: np.ndarray,
    side: str,
    config: Stage1bDecoderConfig,
) -> np.ndarray:
    satellite = remote_satellite(probability, side=side, config=config)
    if config.mode == "remote_satellite":
        return satellite.astype(np.uint8)
    if config.mode == "hybrid":
        local = fixed_local_hysteresis(probability, side=side)
        return np.maximum(local, satellite).astype(np.uint8)
    raise ValueError(f"Unknown Stage-1b mode: {config.mode}")


def apply_stage1b_decoder(
    frame: pd.DataFrame,
    config: Stage1bDecoderConfig,
) -> pd.DataFrame:
    required = {
        "case_id",
        "slice_idx",
        "left_smooth_presence_probability",
        "right_smooth_presence_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing Stage-1b columns: {missing}")

    output = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    for side in SIDES:
        output[f"{side}_smooth_presence"] = 0
        output[f"{side}_span_presence"] = 0

    for _, indices in output.groupby("case_id", sort=False).groups.items():
        ordered = output.loc[list(indices)].sort_values("slice_idx")
        ordered_indices = ordered.index.to_numpy(dtype=int)
        for side in SIDES:
            probability = ordered[f"{side}_smooth_presence_probability"].to_numpy(dtype=float)
            present = decode_stage1b_side(probability, side=side, config=config)
            output.loc[ordered_indices, f"{side}_smooth_presence"] = present
            output.loc[ordered_indices, f"{side}_span_presence"] = present

    for side in SIDES:
        output[f"{side}_smooth_presence"] = output[f"{side}_smooth_presence"].astype(np.uint8)
        output[f"{side}_span_presence"] = output[f"{side}_span_presence"].astype(np.uint8)
    return output
