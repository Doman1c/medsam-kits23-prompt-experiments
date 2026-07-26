from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_candidate(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Candidate must be NAME=/path/to/medsam_validation_summary.json")
    name, path = text.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibrated checkpoint candidates on validation MedSAM Dice")
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for name, path in args.candidate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = payload["recommended_by_validation_medsam"]
        calibration = payload["calibration"]
        rows.append({
            "candidate": name,
            **calibration,
            "expansion_scale": best["expansion_scale"],
            "dice_mean": best["overall"]["dice_mean"],
            "dice_median": best["overall"]["dice_median"],
            "case_level_dice_mean": best["case_level_dice_mean"],
            "oracle_gap_overall": best["oracle_gap_overall"],
            "oracle_gap_case_level": best["oracle_gap_case_level"],
            "small_00_10_dice": best["by_combined_area_bin"]["00_10"]["dice_mean"],
            "medium_10_50_dice": best["by_combined_area_bin"]["10_50"]["dice_mean"],
            "large_50_90_dice": best["by_combined_area_bin"]["50_90"]["dice_mean"],
            "largest_90_100_dice": best["by_combined_area_bin"]["90_100"]["dice_mean"],
        })
    result = pd.DataFrame(rows).sort_values(
        ["case_level_dice_mean", "dice_mean", "oracle_gap_case_level"],
        ascending=[False, False, True],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "candidate_comparison.csv", index=False)
    recommendation = result.iloc[0].to_dict()
    (output_dir / "recommended_final_validation_config.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print("\nRecommended candidate/configuration:")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
