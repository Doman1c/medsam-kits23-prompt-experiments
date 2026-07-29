from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a2.metrics import evaluate_prediction_frame, write_evaluation
from v1a2.utils import parse_int_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dual-kidney prompt quality")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["raw", "smooth", "both"], default="both")
    parser.add_argument("--expansion-scale", type=float, default=1.15)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    variants = ("raw", "smooth") if args.variant == "both" else (args.variant,)
    summaries = {}
    for variant in variants:
        summary, details = evaluate_prediction_frame(
            frame,
            variant=variant,
            expansion_scale=args.expansion_scale,
            target_labels=parse_int_list(args.target_labels),
            presence_threshold=args.presence_threshold,
        )
        write_evaluation(summary, details, args.output_dir, prefix=variant)
        summaries[variant] = summary
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
