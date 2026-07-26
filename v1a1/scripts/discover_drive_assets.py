from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose_archive(project_dir: Path) -> Path:
    preferred_names = [
        "kits23_selected_147_cases.tar",
        "kits23_selected_147_cases.tar.gz",
        "kits23_selected_147_cases.tgz",
    ]
    for name in preferred_names:
        matches = list(project_dir.rglob(name))
        if matches:
            return sorted(matches)[0]
    candidates = []
    for pattern in ("*.tar", "*.tar.gz", "*.tgz", "*.zip"):
        candidates.extend(project_dir.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No KiTS23 archive found under {project_dir}")
    return max(candidates, key=lambda path: path.stat().st_size)


def choose_checkpoint(drive_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {path}")
        return path
    preferred = drive_root / "medsam_vit_b.pth"
    if preferred.exists():
        return preferred
    root_candidates = sorted(drive_root.glob("*.pth"))
    for candidate in root_candidates:
        if "medsam" in candidate.name.lower():
            return candidate
    recursive = sorted(drive_root.rglob("medsam_vit_b.pth"))
    if recursive:
        return recursive[0]
    raise FileNotFoundError(
        f"Could not find medsam_vit_b.pth in the root or subfolders of {drive_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover v1a.1 Drive assets")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--project-dir", default="/content/drive/MyDrive/medsam-kits23")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    project_dir = Path(args.project_dir)
    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive root not found: {drive_root}. Mount Drive first with drive.mount('/content/drive')."
        )
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    payload = {
        "drive_root": str(drive_root.resolve()),
        "project_dir": str(project_dir.resolve()),
        "data_archive": str(choose_archive(project_dir).resolve()),
        "medsam_checkpoint": str(choose_checkpoint(drive_root, args.checkpoint).resolve()),
        "processed_cache": str((project_dir / "cache" / "v1a1_processed_147_cases.tar").resolve()),
        "results_dir": str((project_dir / "v1a1_results").resolve()),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
