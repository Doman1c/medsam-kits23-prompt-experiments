from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME::SUMMARY_JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for item in args.model:
        name, path_text = item.split("::", 1)
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
        overall = payload["overall_decomposition"]
        rows.append({
            "model": name,
            "expansion_scale": float(payload["expansion_scale"]),
            "M0": overall["M0_auto_center_auto_size_mean"],
            "M1": overall["M1_oracle_center_auto_size_mean"],
            "M2": overall["M2_auto_center_oracle_size_mean"],
            "M3": overall["M3_oracle_center_oracle_size_mean"],
            "center_gain": overall["oracle_center_gain_mean_M1_minus_M0"],
            "size_gain": overall["oracle_size_gain_mean_M2_minus_M0"],
            "joint_box_gap": overall["joint_oracle_box_gain_mean_M3_minus_M0"],
            "interaction": overall["center_size_interaction_mean"],
        })

    frame = pd.DataFrame(rows)
    baseline = frame.iloc[0]
    for column in ("M0", "M1", "M2", "M3", "center_gain", "size_gain", "joint_box_gap", "interaction"):
        frame[f"delta_{column}_vs_{baseline['model']}"] = frame[column] - baseline[column]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "fixed_scale_decomposition_comparison.csv", index=False)
    payload = {
        "status": "fixed_expansion_center_size_comparison",
        "development_split": "validation",
        "test_set_used": False,
        "expansion_scale": 1.20,
        "models": frame.to_dict(orient="records"),
        "best_M0": frame.loc[frame["M0"].idxmax(), "model"],
        "lowest_joint_box_gap": frame.loc[frame["joint_box_gap"].idxmin(), "model"],
        "lowest_interaction": frame.loc[frame["interaction"].idxmin(), "model"],
    }
    (output_dir / "fixed_scale_decomposition_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
