from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package compact V1a.3-B validation-stage review files")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    if not root.is_dir():
        raise FileNotFoundError(root)

    stage = output.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    if output.exists():
        output.unlink()
    stage.mkdir(parents=True)

    exact_files = [
        root / "topology_audit" / "topology_summary.json",
        root / "topology_audit" / "topology_case_side.csv",
        root / "topology_audit" / "false_positive_segments.csv",
        root / "span_sweep" / "sweep_summary.json",
        root / "span_sweep" / "b0_locked_baseline_metrics.csv",
        root / "span_sweep" / "span_decoder_sweep.csv",
        root / "span_sweep" / "shortlist_prompt_comparison.csv",
        root / "span_sweep" / "candidate_manifest.json",
    ]
    for source in exact_files:
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(root)
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    candidates_root = root / "span_sweep" / "candidates"
    if not candidates_root.is_dir():
        raise FileNotFoundError(candidates_root)
    for candidate_dir in sorted(path for path in candidates_root.iterdir() if path.is_dir()):
        for relative_name in (
            Path("decoder_config.json"),
            Path("recommended_calibration.json"),
            Path("prompt_metrics/smooth_summary.json"),
        ):
            source = candidate_dir / relative_name
            if not source.is_file():
                raise FileNotFoundError(source)
            relative = source.relative_to(root)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    negative_root = root / "negative_aware_medsam"
    if negative_root.is_dir():
        for name in ("negative_aware_summary.json", "negative_aware_candidate_comparison.csv"):
            source = negative_root / name
            if source.is_file():
                relative = source.relative_to(root)
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(stage)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "bundle_type": "v1a3b_validation_stage_review",
        "test_results": False,
        "contains_weights": False,
        "contains_npz": False,
        "contains_large_prediction_tables": False,
        "files": files,
    }
    (stage / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage.parent))

    print(f"Created: {output}")
    print(f"Files: {len(files)}")
    print(f"Size MiB: {output.stat().st_size / 1024 / 1024:.2f}")
    print(f"SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()
