from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a3b.span_decoder import SIDES, SpanDecoderConfig, apply_span_decoder
from v1a1.calibration import add_threshold_independent_smoothing


B0 = SpanDecoderConfig(
    mode="baseline",
    left_threshold=0.005,
    right_threshold=0.005,
    baseline_max_gap=2,
)

B1 = SpanDecoderConfig(
    mode="single_span",
    left_threshold=0.0035,
    right_threshold=0.04,
    baseline_max_gap=4,
    max_internal_gap=2,
    min_span_length=1,
    span_penalty=0.0,
    length_penalty=0.0,
)

BOUNDARY_PAD = 1
EXPANSION_SCALE = 1.20


def validate_test_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    test = manifest[manifest["split"].astype(str) == "test"].copy()
    if int(test["case_id"].nunique()) != 27 or len(test) != 4277:
        raise RuntimeError(
            f"Unexpected test split: cases={test['case_id'].nunique()}, rows={len(test)}"
        )
    frame_keys = list(
        zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int))
    )
    test_keys = list(
        zip(test["case_id"].astype(str), test["slice_idx"].astype(int))
    )
    if sorted(frame_keys) != sorted(test_keys):
        raise RuntimeError(
            "Prediction rows do not exactly match the frozen 27-case test split."
        )


def pad_single_span_predictions(frame: pd.DataFrame, boundary_pad: int) -> pd.DataFrame:
    pad = int(boundary_pad)
    if pad < 0:
        raise ValueError("boundary_pad must be non-negative")
    output = frame.sort_values(["case_id", "slice_idx"]).reset_index(drop=True).copy()
    if pad == 0:
        return output

    for _, indices in output.groupby("case_id", sort=False).groups.items():
        ordered = output.loc[list(indices)].sort_values("slice_idx")
        ordered_indices = ordered.index.to_numpy(dtype=int)
        for side in SIDES:
            present = ordered[f"{side}_smooth_presence"].to_numpy(dtype=np.uint8).astype(bool)
            positive = np.flatnonzero(present)
            if not len(positive):
                continue
            start = max(int(positive[0]) - pad, 0)
            end = min(int(positive[-1]) + pad, len(present) - 1)
            padded = np.zeros(len(present), dtype=np.uint8)
            padded[start : end + 1] = 1
            output.loc[ordered_indices, f"{side}_smooth_presence"] = padded
            output.loc[ordered_indices, f"{side}_span_presence"] = padded

    for side in SIDES:
        output[f"{side}_smooth_presence"] = output[f"{side}_smooth_presence"].astype(np.uint8)
        output[f"{side}_span_presence"] = output[f"{side}_span_presence"].astype(np.uint8)
    return output


def write_candidate(
    output_dir: Path,
    name: str,
    frame: pd.DataFrame,
    decoder: dict,
) -> dict:
    candidate_dir = output_dir / "candidates" / name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    predictions = candidate_dir / "predictions.csv"
    calibration = candidate_dir / "calibration.json"
    decoder_path = candidate_dir / "decoder_config.json"

    frame.to_csv(predictions, index=False)

    calibration_payload = {
        "configuration": {
            "variant": "smooth",
            "max_gap": 0,
            "left_threshold": float(decoder["left_threshold"]),
            "right_threshold": float(decoder["right_threshold"]),
        },
        "decoder": decoder,
        "status": "v1a3b_frozen_secondary_test_materialized_predictions",
        "selection_performed_on_test": False,
        "test_set_used": True,
    }
    calibration.write_text(
        json.dumps(calibration_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    decoder_path.write_text(
        json.dumps(
            {
                "name": name,
                "decoder": decoder,
                "expansion_scale": EXPANSION_SCALE,
                "selection_performed_on_test": False,
                "test_set_used": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "name": name,
        "predictions": str(predictions),
        "calibration_json": str(calibration),
        "expansion_scale": EXPANSION_SCALE,
        "decoder": decoder,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize locked V1a.3-B B0/B1 predictions on the previously exposed test split."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    frame = pd.read_csv(prediction_path, dtype={"case_id": str})
    validate_test_rows(frame, manifest_path)
    frame = add_threshold_independent_smoothing(frame)

    b0 = apply_span_decoder(frame, B0)

    b1 = apply_span_decoder(frame, B1)
    b1 = pad_single_span_predictions(b1, boundary_pad=BOUNDARY_PAD)

    output_dir.mkdir(parents=True, exist_ok=True)

    b0_decoder = B0.to_dict()
    b0_decoder["boundary_pad"] = 0

    b1_decoder = B1.to_dict()
    b1_decoder["boundary_pad"] = BOUNDARY_PAD

    candidates = [
        write_candidate(output_dir, "b0_locked_baseline", b0, b0_decoder),
        write_candidate(output_dir, "b1_low_fpr_frozen", b1, b1_decoder),
    ]

    manifest_payload = {
        "status": "v1a3b_locked_secondary_test_candidates",
        "selection_performed_on_test": False,
        "test_set_used": True,
        "source_predictions": str(prediction_path),
        "candidates": candidates,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
