from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone MedSAM and copy checkpoint")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--checkpoint-dest", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/bowang-lab/MedSAM.git", str(repo_dir)],
            check=True,
        )
    checkpoint_source = Path(args.checkpoint_source)
    if not checkpoint_source.exists():
        raise FileNotFoundError(checkpoint_source)
    checkpoint_dest = Path(args.checkpoint_dest)
    checkpoint_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_source, checkpoint_dest)
    print(f"MedSAM repo: {repo_dir}")
    print(f"Checkpoint: {checkpoint_dest}")


if __name__ == "__main__":
    main()
