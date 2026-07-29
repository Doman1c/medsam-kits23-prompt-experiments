from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    negative_csv = root / "01_negative_aware" / "negative_aware_candidate_comparison.csv"
    fixed_csv = root / "02_fixed_scale" / "comparison" / "fixed_scale_decomposition_comparison.csv"
    checkpoint_csv = root / "03_checkpoint_pareto" / "comparison" / "checkpoint_pareto_comparison.csv"

    for path in (negative_csv, fixed_csv, checkpoint_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    negative = pd.read_csv(negative_csv)
    fixed = pd.read_csv(fixed_csv)
    checkpoints = pd.read_csv(checkpoint_csv)

    report_dir = root / "final_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "v1a2_followup_validation_complete",
        "development_split": "validation",
        "test_set_used": False,
        "negative_aware_best_case_volume": negative.sort_values(
            ["case_volume_dice_mean", "all_slice_dice_mean", "negative_false_positive_mask_rate"],
            ascending=[False, False, True],
        ).iloc[0].to_dict(),
        "fixed_scale_best_M0": fixed.loc[fixed["M0"].idxmax()].to_dict(),
        "fixed_scale_lowest_interaction": fixed.loc[fixed["interaction"].idxmin()].to_dict(),
        "checkpoint_pareto_winner": checkpoints.iloc[0].to_dict(),
    }
    (report_dir / "followup_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = [
        "# V1a.2 follow-up validation", "",
        "## 1. Negative-aware all-slice evaluation", "",
        negative.to_markdown(index=False, floatfmt=".6f"), "",
        "## 2. Fixed expansion = 1.20 decomposition", "",
        fixed.to_markdown(index=False, floatfmt=".6f"), "",
        "## 3. V1a.2-B checkpoint Pareto", "",
        checkpoints.to_markdown(index=False, floatfmt=".6f"), "",
        "## Machine-readable summary", "", "```json",
        json.dumps(payload, indent=2, ensure_ascii=False), "```",
    ]
    (report_dir / "followup_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
