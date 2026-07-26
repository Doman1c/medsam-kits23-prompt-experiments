from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


def safe_extract_tar(source: Path, destination: Path) -> None:
    with tarfile.open(source, "r:*") as archive:
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            root = destination.resolve()
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
            archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy or extract KiTS23 data into local Colab storage")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() and args.skip_existing and any(destination.iterdir()):
        print(f"Using existing destination: {destination}")
    else:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for item in source.iterdir():
                target = destination / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        elif tarfile.is_tarfile(source):
            safe_extract_tar(source, destination)
        elif zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                archive.extractall(destination)
        else:
            raise ValueError(f"Unsupported data source: {source}")

    npz_count = sum(1 for _ in destination.rglob("*.npz"))
    nifti_count = sum(1 for _ in destination.rglob("*.nii.gz"))
    print(f"Data synchronized to: {destination}")
    print(f"Detected .npz files: {npz_count}")
    print(f"Detected .nii.gz files: {nifti_count}")


if __name__ == "__main__":
    main()
