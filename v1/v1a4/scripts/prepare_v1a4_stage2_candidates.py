from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

SELECTED = [
    "v1a3b_frozen_b1",
    "stage1_fixed_local_hysteresis",
    "r2b_remote_satellite_1",
    "r3_hybrid_1",
]

EXPECTED = {
    "r2b_remote_satellite_1": {
        "mode": "remote_satellite",
        "satellite_max_distance": 64,
        "satellite_evidence_ratio": 0.1,
        "satellite_min_length": 1,
        "max_satellites": 1,
    },
    "r3_hybrid_1": {
        "mode": "hybrid",
        "satellite_max_distance": 64,
        "satellite_evidence_ratio": 0.1,
        "satellite_min_length": 1,
        "max_satellites": 1,
    },
}


def assert_close(actual: float, expected: float, name: str) -> None:
    if abs(float(actual) - float(expected)) > 1e-12:
        raise RuntimeError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the predeclared V1a.4 Stage-2 MedSAM candidate set"
    )
    parser.add_argument("--stage1b-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    args = parser.parse_args()

    stage1b_root = Path(args.stage1b_root)
    source_manifest_path = stage1b_root / "candidate_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("test_set_used") is not False:
        raise RuntimeError("Stage-1b source is not validation-only")

    by_name = {item["name"]: item for item in source_manifest["candidates"]}
    missing = [name for name in SELECTED if name not in by_name]
    if missing:
        raise RuntimeError(f"Missing predeclared Stage-2 candidates: {missing}")

    # Lock the two Stage-1b finalists before any MedSAM output exists.
    for name, expected in EXPECTED.items():
        item = by_name[name]
        decoder = item["decoder"]
        for key, value in expected.items():
            if isinstance(value, float):
                assert_close(decoder[key], value, f"{name}.{key}")
            elif decoder[key] != value:
                raise RuntimeError(
                    f"{name}.{key}: expected {value}, got {decoder[key]}"
                )
        presence = item["presence_metrics"]
        if not (
            float(presence["positive_recall"]) >= 0.96
            and float(presence["left_positive_recall"]) >= 0.94
            and float(presence["right_positive_recall"]) >= 0.97
            and float(presence["false_positive_rate"]) <= 0.15
        ):
            raise RuntimeError(f"{name} no longer satisfies the locked V1a.4 gate")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    candidate_specs = []
    rows = []
    for name in SELECTED:
        item = by_name[name]
        source_dir = Path(item["directory"])
        required = [
            source_dir / "recommended_predictions.csv",
            source_dir / "recommended_calibration.json",
            source_dir / "decoder_config.json",
            source_dir / "topology_case_side.csv",
            source_dir / "false_positive_segments.csv",
        ]
        for source in required:
            if not source.is_file():
                raise FileNotFoundError(source)

        target_dir = output_dir / "candidates" / name
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in required:
            shutil.copy2(source, target_dir / source.name)

        copied_config = json.loads(
            (target_dir / "decoder_config.json").read_text(encoding="utf-8")
        )
        presence = copied_config["presence_metrics"]
        decoder = copied_config["decoder"]

        candidate_specs.append(
            {
                "name": name,
                "directory": str(target_dir),
                "predictions": str(target_dir / "recommended_predictions.csv"),
                "calibration_json": str(target_dir / "recommended_calibration.json"),
                "expansion_scale": float(args.expansion_scale),
                "decoder": decoder,
                "presence_metrics": presence,
            }
        )
        rows.append(
            {
                "name": name,
                "mode": decoder["mode"],
                "positive_recall": presence["positive_recall"],
                "left_positive_recall": presence["left_positive_recall"],
                "right_positive_recall": presence["right_positive_recall"],
                "false_positive_rate": presence["false_positive_rate"],
                "specificity": presence["specificity"],
                "satellite_max_distance": decoder.get("satellite_max_distance", 0),
                "satellite_evidence_ratio": decoder.get("satellite_evidence_ratio", 0.0),
                "satellite_min_length": decoder.get("satellite_min_length", 1),
            }
        )

    manifest = {
        "status": "validation_only_v1a4_stage2_predeclared_candidates",
        "test_set_used": False,
        "selection_performed_before_medsam": True,
        "selection_rule": (
            "Compare frozen B1 and the fixed Stage-1 local hysteresis reference against "
            "the first predeclared feasible remote-satellite and hybrid candidates from "
            "Stage 1b. No MedSAM result is used to select these four inputs."
        ),
        "candidates": candidate_specs,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output_dir / "stage2_presence_candidates.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Prepared {len(rows)} V1a.4 Stage-2 candidates in {output_dir}")


if __name__ == "__main__":
    main()
