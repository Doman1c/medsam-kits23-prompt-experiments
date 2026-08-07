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


def copy_file(source: Path, root: Path, stage: Path) -> None:
    destination = stage / source.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package compact V1a.3-B Stage-2 MedSAM review files")
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

    required = [
        root / "candidate_set" / "candidate_manifest.json",
        root / "candidate_set" / "stage2_presence_candidates.csv",
        root / "negative_aware_medsam" / "negative_aware_summary.json",
        root / "negative_aware_medsam" / "negative_aware_candidate_comparison.csv",
        root / "stage2_combined_comparison.csv",
    ]
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)
        copy_file(source, root, stage)

    candidates_root = root / "candidate_set" / "candidates"
    if not candidates_root.is_dir():
        raise FileNotFoundError(candidates_root)
    for candidate_dir in sorted(path for path in candidates_root.iterdir() if path.is_dir()):
        for relative_name in (
            Path("decoder_config.json"),
            Path("recommended_calibration.json"),
            Path("topology_case_side.csv"),
            Path("false_positive_segments.csv"),
            Path("prompt_metrics/smooth_summary.json"),
        ):
            source = candidate_dir / relative_name
            if not source.is_file():
                raise FileNotFoundError(source)
            copy_file(source, root, stage)

    negative_root = root / "negative_aware_medsam"
    for source in sorted(negative_root.rglob("*")):
        if not source.is_file():
            continue
        if source.suffix.lower() not in {".json", ".csv", ".txt"}:
            continue
        if source.stat().st_size > 5 * 1024 * 1024:
            continue
        destination = stage / source.relative_to(root)
        if not destination.exists():
            copy_file(source, root, stage)

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

    (stage / "submission_manifest.json").write_text(
        json.dumps(
            {
                "bundle_type": "v1a3b_stage2_medsam_review",
                "test_results": False,
                "contains_weights": False,
                "contains_npz": False,
                "contains_large_prediction_tables": False,
                "files": files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
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
