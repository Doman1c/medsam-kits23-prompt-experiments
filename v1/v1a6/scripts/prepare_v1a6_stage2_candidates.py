from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

SELECTED = ["g0_frozen_b1", "g1_r2b_native", "g6_robust_linear"]


def copy_required(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("recommended_predictions.csv", "recommended_calibration.json"):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target_dir / name)

    config = source_dir / "decoder_config.json"
    if config.is_file():
        shutil.copy2(config, target_dir / config.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen V1a.6 Stage-2 candidates")
    parser.add_argument("--stage1-root", required=True)
    parser.add_argument("--b1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expansion-scale", type=float, default=1.20)
    args = parser.parse_args()

    stage1_root = Path(args.stage1_root)
    b1_dir = Path(args.b1_dir)
    stage1_manifest_path = stage1_root / "candidate_set" / "candidate_manifest.json"
    stage1_gate_path = stage1_root / "stage1_geometry_gate.csv"
    if not stage1_manifest_path.is_file():
        raise FileNotFoundError(stage1_manifest_path)
    if not stage1_gate_path.is_file():
        raise FileNotFoundError(stage1_gate_path)

    stage1_manifest = json.loads(stage1_manifest_path.read_text(encoding="utf-8"))
    if stage1_manifest.get("test_set_used") is not False:
        raise RuntimeError("V1a.6 Stage-1 source is not validation-only")
    if int(stage1_manifest.get("validation_cases", -1)) != 20 or int(stage1_manifest.get("validation_rows", -1)) != 3644:
        raise RuntimeError("Unexpected V1a.6 Stage-1 validation split")

    by_name = {item["name"]: item for item in stage1_manifest["candidates"]}
    for name in ("g1_r2b_native", "g6_robust_linear"):
        if name not in by_name:
            raise RuntimeError(f"Missing Stage-1 candidate: {name}")

    gate = pd.read_csv(stage1_gate_path)
    g6 = gate[gate["name"] == "g6_robust_linear"]
    if len(g6) != 1 or not bool(g6.iloc[0]["advance_to_medsam_review"]):
        raise RuntimeError("G6 no longer passes the frozen Stage-1 geometry gate")

    g6_manifest = by_name["g6_robust_linear"]
    inv = g6_manifest["invariants"]
    if inv.get("presence_identical_to_r2b") is not True:
        raise RuntimeError("G6 presence invariance failed")
    if int(inv.get("changed_non_recovered_geometry_values", -1)) != 0:
        raise RuntimeError("G6 changed non-recovered geometry")
    decoder = g6_manifest["decoder"]
    expected = {
        "mode": "robust_linear",
        "mad_threshold": 3.5,
        "minimum_inliers": 5,
    }
    for key, value in expected.items():
        if decoder.get(key) != value:
            raise RuntimeError(f"Unexpected G6 decoder {key}: {decoder.get(key)}")

    source_dirs = {
        "g0_frozen_b1": b1_dir,
        "g1_r2b_native": stage1_root / "candidate_set" / "candidates" / "g1_r2b_native",
        "g6_robust_linear": stage1_root / "candidate_set" / "candidates" / "g6_robust_linear",
    }

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    candidates = []
    rows = []
    for name in SELECTED:
        source_dir = source_dirs[name]
        target_dir = output_dir / "candidates" / name
        copy_required(source_dir, target_dir)
        candidates.append(
            {
                "name": name,
                "directory": str(target_dir),
                "predictions": str(target_dir / "recommended_predictions.csv"),
                "calibration_json": str(target_dir / "recommended_calibration.json"),
                "expansion_scale": float(args.expansion_scale),
            }
        )
        rows.append(
            {
                "name": name,
                "role": {
                    "g0_frozen_b1": "complete_system_reference",
                    "g1_r2b_native": "r2b_native_geometry_reference",
                    "g6_robust_linear": "stage1_geometry_gate_winner",
                }[name],
                "expansion_scale": float(args.expansion_scale),
            }
        )

    payload = {
        "status": "validation_only_v1a6_stage2_predeclared_candidates",
        "test_set_used": False,
        "selection_performed_before_medsam": True,
        "selection_rule": (
            "Frozen B1 complete-system reference, frozen R2b native-geometry reference, "
            "and G6, the only V1a.6 Stage-1 method passing both recovered-IoU and "
            "recovered-center-error gates."
        ),
        "candidates": candidates,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output_dir / "stage2_candidates.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Prepared {len(rows)} V1a.6 Stage-2 candidates in {output_dir}")


if __name__ == "__main__":
    main()
