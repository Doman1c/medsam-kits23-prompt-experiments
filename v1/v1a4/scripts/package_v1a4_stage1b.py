from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ALLOWED_SUFFIXES = {".json", ".csv", ".txt"}
EXCLUDED_NAMES = {
    "recommended_predictions.csv",
    "topology_case_side.csv",
    "false_positive_segments.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package compact V1a.4 Stage-1b review results")
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

    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        if source.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if source.name in EXCLUDED_NAMES:
            continue
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
        "bundle_type": "v1a4_stage1b_validation_review",
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

    print("Created:", output)
    print("Files:", len(files))
    print("Size MiB:", round(output.stat().st_size / 1024 / 1024, 3))
    print("SHA256:", sha256(output))


if __name__ == "__main__":
    main()
