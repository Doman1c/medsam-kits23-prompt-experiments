from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_model(text: str) -> tuple[str, Path, Path]:
    parts = text.split("::")
    if len(parts) != 3:
        raise ValueError("--model format must be NAME::CONFIG_JSON::DECOMPOSITION_JSON")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V1a.1, V1a.2-A calibration line, and V1a.2-B.")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.model:
        name, config_path, dec_path = parse_model(spec)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dec = json.loads(dec_path.read_text(encoding="utf-8"))["overall_decomposition"]
        rows.append({
            "model": name,
            "slice_dice": config["dice_mean"],
            "case_dice": config["case_level_dice_mean"],
            "oracle_gap": config["oracle_gap_overall"],
            "dice_00_10": config["small_00_10_dice"],
            "dice_10_50": config["medium_10_50_dice"],
            "dice_50_90": config["large_50_90_dice"],
            "dice_90_100": config["largest_90_100_dice"],
            "M0": dec["M0_auto_center_auto_size_mean"],
            "M1": dec["M1_oracle_center_auto_size_mean"],
            "M2": dec["M2_auto_center_oracle_size_mean"],
            "M3": dec["M3_oracle_center_oracle_size_mean"],
            "center_gain": dec["oracle_center_gain_mean_M1_minus_M0"],
            "size_gain": dec["oracle_size_gain_mean_M2_minus_M0"],
            "joint_box_gap": dec["joint_oracle_box_gain_mean_M3_minus_M0"],
            "interaction": dec["center_size_interaction_mean"],
            "presence_candidate": config.get("presence_candidate"),
            "variant": config.get("variant"),
            "max_gap": config.get("max_gap"),
            "left_threshold": config.get("left_threshold"),
            "right_threshold": config.get("right_threshold"),
            "expansion_scale": config.get("expansion_scale"),
        })

    frame = pd.DataFrame(rows)
    baseline = frame.iloc[0]
    for column in (
        "slice_dice", "case_dice", "oracle_gap", "dice_00_10", "dice_10_50",
        "dice_50_90", "dice_90_100", "M0", "center_gain", "size_gain",
        "joint_box_gap", "interaction",
    ):
        frame[f"delta_{column}_vs_{baseline['model']}"] = frame[column] - baseline[column]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "all_model_validation_comparison.csv", index=False)

    summary = {
        "models": frame.to_dict(orient="records"),
        "descriptive_winners": {
            "slice_dice": frame.loc[frame["slice_dice"].idxmax(), "model"],
            "case_dice": frame.loc[frame["case_dice"].idxmax(), "model"],
            "M0_geometry": frame.loc[frame["M0"].idxmax(), "model"],
            "lowest_interaction": frame.loc[frame["interaction"].idxmin(), "model"],
        },
        "decision_notes": "End-to-end claims require slice and case Dice; geometry claims use M0, joint gap and interaction. Do not use test data to revise these choices.",
    }
    (output_dir / "all_model_validation_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "all_model_validation_comparison.md").write_text(
        "\n".join([
            "# Validation comparison", "", frame.to_markdown(index=False, floatfmt=".6f"), "",
            "## Descriptive winners", "", "```json",
            json.dumps(summary["descriptive_winners"], indent=2, ensure_ascii=False), "```",
        ]),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print("\nDescriptive winners:")
    print(json.dumps(summary["descriptive_winners"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
