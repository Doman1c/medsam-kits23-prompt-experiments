from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the presence candidate by full MedSAM validation.")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.candidate_manifest).read_text(encoding="utf-8"))["candidates"]
    rows = []
    for candidate in manifest:
        summary_path = Path(candidate["directory"]) / "medsam" / "medsam_validation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        best = payload["recommended_by_validation_medsam"]
        rows.append({
            **candidate,
            "medsam_summary": str(summary_path),
            "expansion_scale": float(best["expansion_scale"]),
            "dice_mean": float(best["overall"]["dice_mean"]),
            "dice_median": float(best["overall"]["dice_median"]),
            "case_level_dice_mean": float(best["case_level_dice_mean"]),
            "oracle_gap_overall": float(best["oracle_gap_overall"]),
            "oracle_gap_case_level": float(best["oracle_gap_case_level"]),
        })

    table = pd.DataFrame(rows).sort_values(
        ["case_level_dice_mean", "dice_mean", "false_positive_rate"],
        ascending=[False, False, True],
    )
    winner = table.iloc[0].to_dict()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "presence_candidate_medsam_table.csv", index=False)
    (output_dir / "presence_candidate_winner.json").write_text(
        json.dumps({
            "winner": winner,
            "selection_rule": "Maximize validation case-level MedSAM Dice, break ties with slice-level Dice, then lower FPR.",
            "all_candidates": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print("\nWinner:")
    print(json.dumps(winner, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
