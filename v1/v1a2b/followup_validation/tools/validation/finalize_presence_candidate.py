from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the selected presence candidate and build a locked validation configuration.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--winner-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    winner_payload = json.loads(Path(args.winner_json).read_text(encoding="utf-8"))
    winner = winner_payload["winner"]
    source_dir = Path(winner["directory"])
    calibration_path = Path(winner["calibration_json"])
    predictions_path = Path(winner["predictions_csv"])
    medsam_summary_path = Path(winner["medsam_summary"])
    checkpoint_path = Path(args.checkpoint)

    medsam_payload = json.loads(medsam_summary_path.read_text(encoding="utf-8"))
    selected = medsam_payload["recommended_by_validation_medsam"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calibration_path, output_dir / "recommended_calibration.json")
    shutil.copy2(predictions_path, output_dir / "recommended_predictions.csv")
    shutil.copytree(source_dir / "medsam", output_dir / "medsam", dirs_exist_ok=True)

    area = selected["by_combined_area_bin"]
    config = {
        "status": "locked_validation_configuration",
        "development_split": "validation",
        "test_set_used": False,
        "experiment": args.experiment,
        "presence_candidate": winner["name"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "weights": "ema",
        "variant": winner["variant"],
        "max_gap": int(winner["max_gap"]),
        "left_threshold": float(winner["left_threshold"]),
        "right_threshold": float(winner["right_threshold"]),
        "expansion_scale": float(selected["expansion_scale"]),
        "dice_mean": float(selected["overall"]["dice_mean"]),
        "dice_median": float(selected["overall"]["dice_median"]),
        "case_level_dice_mean": float(selected["case_level_dice_mean"]),
        "oracle_scaled_dice_mean": float(selected["oracle_scaled_overall"]["dice_mean"]),
        "oracle_gap_overall": float(selected["oracle_gap_overall"]),
        "oracle_gap_case_level": float(selected["oracle_gap_case_level"]),
        "small_00_10_dice": float(area["00_10"]["dice_mean"]),
        "medium_10_50_dice": float(area["10_50"]["dice_mean"]),
        "large_50_90_dice": float(area["50_90"]["dice_mean"]),
        "largest_90_100_dice": float(area["90_100"]["dice_mean"]),
        "presence_metrics": {
            "positive_recall": float(winner["positive_recall"]),
            "left_positive_recall": float(winner["left_positive_recall"]),
            "right_positive_recall": float(winner["right_positive_recall"]),
            "false_positive_rate": float(winner["false_positive_rate"]),
            "prompt_success_rate": float(winner["prompt_success_rate"]),
            "continuous_score": float(winner["continuous_score"]),
        },
        "selection_rule": winner_payload["selection_rule"],
        "artifact_sha256": {
            "checkpoint": sha256(checkpoint_path),
            "calibration": sha256(calibration_path),
            "predictions": sha256(predictions_path),
            "medsam_summary": sha256(medsam_summary_path),
        },
        "research_guardrail": "Do not use the frozen test set for checkpoint or calibration selection.",
    }
    (output_dir / "recommended_final_validation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
