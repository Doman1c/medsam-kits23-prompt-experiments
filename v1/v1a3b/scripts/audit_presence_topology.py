from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "v1" / "v1a1"))

from v1a3b.span_decoder import SpanDecoderConfig, apply_span_decoder, summarize_presence_topology
from v1a1.calibration import add_threshold_independent_smoothing


def validate_validation_rows(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    val = manifest[manifest["split"].astype(str) == "val"].copy()
    val["slice_idx"] = val["slice_idx"].astype(int)
    frame_keys = set(zip(frame["case_id"].astype(str), frame["slice_idx"].astype(int)))
    val_keys = set(zip(val["case_id"].astype(str), val["slice_idx"].astype(int)))
    if frame_keys != val_keys:
        missing = sorted(val_keys - frame_keys)[:10]
        extra = sorted(frame_keys - val_keys)[:10]
        raise RuntimeError(
            "V1a.3-B development is validation-only and prediction keys must exactly match the val split. "
            f"Missing={missing}; extra={extra}"
        )
    if frame["case_id"].astype(str).nunique() != val["case_id"].astype(str).nunique():
        raise RuntimeError("Validation case-count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V1a.3-A validation presence topology before span decoding")
    parser.add_argument("--predictions", required=True, help="V1a.3-A epoch-18 base validation predictions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--left-threshold", type=float, default=0.005)
    parser.add_argument("--right-threshold", type=float, default=0.005)
    parser.add_argument("--max-gap", type=int, default=2)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    frame = pd.read_csv(prediction_path, dtype={"case_id": str})
    validate_validation_rows(frame, manifest_path)
    frame = add_threshold_independent_smoothing(frame)
    config = SpanDecoderConfig(
        mode="baseline",
        left_threshold=args.left_threshold,
        right_threshold=args.right_threshold,
        baseline_max_gap=args.max_gap,
    )
    decoded = apply_span_decoder(frame, config)
    summary, case_side, false_segments = summarize_presence_topology(decoded)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(output_dir / "baseline_decoded_predictions.csv", index=False)
    case_side.to_csv(output_dir / "topology_case_side.csv", index=False)
    false_segments.to_csv(output_dir / "false_positive_segments.csv", index=False)

    payload = {
        "status": "validation_only_presence_topology_audit",
        "test_set_used": False,
        "source_predictions": str(prediction_path),
        "manifest": str(manifest_path),
        "baseline_decoder": config.to_dict(),
        "summary": summary,
    }
    (output_dir / "topology_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
