from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a2b.metrics import evaluate_prediction_frame
from v1a2b.utils import parse_float_list, parse_int_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only expansion sweep for automatic dual boxes")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", default="1.10,1.15,1.20,1.25")
    parser.add_argument("--target-labels", default="1")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in parse_float_list(args.scales):
        for variant in ("raw", "smooth"):
            summary, _ = evaluate_prediction_frame(
                frame,
                variant=variant,
                expansion_scale=scale,
                target_labels=parse_int_list(args.target_labels),
            )
            rows.append(summary)
    result = pd.DataFrame(rows).sort_values(
        ["prompt_success_rate", "continuous_score", "expanded_coverage_mean"],
        ascending=False,
    )
    result.to_csv(output_dir / "expansion_sweep.csv", index=False)
    best = result.iloc[0].to_dict()
    (output_dir / "recommended_prompt_scale.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print("\nPrompt-metric recommendation (MedSAM validation Dice remains the final selector):")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
