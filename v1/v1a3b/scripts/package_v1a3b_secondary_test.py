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


def copy_required(root: Path, stage: Path, relative: str) -> None:
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package compact V1a.3-B locked secondary-test review files."
    )
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

    exact = [
        "FROZEN_LOCK.json",
        "candidate_set/candidate_manifest.json",
        "candidate_set/candidates/b0_locked_baseline/decoder_config.json",
        "candidate_set/candidates/b0_locked_baseline/calibration.json",
        "candidate_set/candidates/b0_locked_baseline/prompt_metrics/smooth_summary.json",
        "candidate_set/candidates/b1_low_fpr_frozen/decoder_config.json",
        "candidate_set/candidates/b1_low_fpr_frozen/calibration.json",
        "candidate_set/candidates/b1_low_fpr_frozen/prompt_metrics/smooth_summary.json",
        "negative_aware_medsam/negative_aware_candidate_comparison.csv",
        "negative_aware_medsam/negative_aware_summary.json",
        "secondary_test_comparison.csv",
        "secondary_test_summary.json",
    ]
    for relative in exact:
        copy_required(root, stage, relative)

    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(stage)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    manifest = {
        "bundle_type": "v1a3b_locked_secondary_test_review",
        "test_results": True,
        "test_is_pristine": False,
        "selection_performed_on_test": False,
        "contains_weights": False,
        "contains_npz": False,
        "contains_large_prediction_tables": False,
        "files": files,
    }
    (stage / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
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
