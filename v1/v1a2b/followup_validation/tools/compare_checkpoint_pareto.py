from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--negative-aware-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", default="4,5,11,14")
    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    epochs = [int(item) for item in args.epochs.split(",") if item.strip()]
    negative_payload = json.loads(Path(args.negative_aware_summary).read_text(encoding="utf-8"))
    negative_by_name = {row["candidate"]: row for row in negative_payload["candidates"]}

    rows = []
    for epoch in epochs:
        name = f"epoch_{epoch:03d}"
        final_config_path = root / name / "final" / "recommended_final_validation_config.json"
        if not final_config_path.is_file():
            raise FileNotFoundError(final_config_path)
        config = json.loads(final_config_path.read_text(encoding="utf-8"))
        negative = negative_by_name[name]
        rows.append({
            "epoch": epoch,
            "checkpoint_candidate": name,
            "presence_candidate": config.get("presence_candidate"),
            "variant": config["variant"],
            "max_gap": config["max_gap"],
            "left_threshold": config["left_threshold"],
            "right_threshold": config["right_threshold"],
            "expansion_scale": config["expansion_scale"],
            "positive_slice_dice": config["dice_mean"],
            "positive_case_dice": config["case_level_dice_mean"],
            "positive_oracle_gap": config["oracle_gap_overall"],
            "presence_recall": config["presence_metrics"]["positive_recall"],
            "presence_fpr_side": config["presence_metrics"]["false_positive_rate"],
            "all_slice_dice": negative["all_slice_dice_mean"],
            "all_slice_case_dice": negative["all_slice_case_mean_dice"],
            "case_volume_dice": negative["case_volume_dice_mean"],
            "negative_specificity": negative["negative_slice_specificity"],
            "negative_fp_mask_rate": negative["negative_false_positive_mask_rate"],
            "negative_prompt_rate": negative["negative_prompt_rate"],
            "negative_predicted_fraction_mean": negative["negative_predicted_fraction_mean"],
        })

    frame = pd.DataFrame(rows).sort_values(
        ["case_volume_dice", "all_slice_dice", "negative_fp_mask_rate"],
        ascending=[False, False, True],
    )
    winner = frame.iloc[0].to_dict()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "checkpoint_pareto_comparison.csv", index=False)
    payload = {
        "status": "validation_only_checkpoint_pareto",
        "test_set_used": False,
        "selection_rule": "Maximize negative-aware case-volume Dice, then all-slice mean Dice, then minimize negative false-positive mask rate.",
        "winner": winner,
        "checkpoints": frame.to_dict(orient="records"),
    }
    (output_dir / "checkpoint_pareto_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print("\nWinner:")
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
