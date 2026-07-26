from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(command: list[str]) -> None:
    print("\n+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *map(str, args)]


def discover_assets(drive_root: Path, project_dir: Path, checkpoint: str | None) -> dict:
    output = Path("/tmp/v1a1_assets.json")
    command = py(
        "discover_drive_assets.py",
        "--drive-root", str(drive_root),
        "--project-dir", str(project_dir),
        "--output-json", str(output),
    )
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    run(command)
    return json.loads(output.read_text(encoding="utf-8"))



CASE_RE = re.compile(r"case_\d+")


def validate_processed_dataset(processed_root: Path, expected_cases: int = 147) -> tuple[bool, dict]:
    files = sorted(processed_root.rglob("*.npz")) if processed_root.exists() else []
    case_ids = set()
    zero_byte = []
    invalid_zip = []
    missing_members = []
    for path in files:
        match = CASE_RE.search(str(path))
        if match:
            case_ids.add(match.group(0))
        if path.stat().st_size == 0:
            zero_byte.append(str(path))
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if not {"image.npy", "label.npy"}.issubset(names):
                    missing_members.append(str(path))
                elif archive.testzip() is not None:
                    invalid_zip.append(str(path))
        except (zipfile.BadZipFile, EOFError, OSError):
            invalid_zip.append(str(path))
    metadata_files = sorted(processed_root.rglob("case_metadata.json")) if processed_root.exists() else []
    expected_slices = 0
    metadata_errors = []
    layout_errors = []
    required_format = "v1a1_dual_yx_v2"
    for meta in metadata_files:
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            expected_slices += int(payload["num_slices"])
            if payload.get("processed_format_version") != required_format:
                layout_errors.append(
                    f"{meta}: expected processed_format_version={required_format}, "
                    f"got {payload.get('processed_format_version')}"
                )
            if payload.get("volume_array_layout") != "ZYX":
                layout_errors.append(
                    f"{meta}: expected volume_array_layout=ZYX, "
                    f"got {payload.get('volume_array_layout')}"
                )
            if payload.get("anatomical_lr_axis_in_slice") != "columns":
                layout_errors.append(
                    f"{meta}: expected anatomical_lr_axis_in_slice=columns, "
                    f"got {payload.get('anatomical_lr_axis_in_slice')}"
                )
        except Exception as exc:
            metadata_errors.append(f"{meta}: {exc}")
    valid = (
        len(case_ids) == expected_cases
        and len(files) > 0
        and not zero_byte
        and not invalid_zip
        and not missing_members
        and not metadata_errors
        and not layout_errors
        and (expected_slices == 0 or expected_slices == len(files))
    )
    report = {
        "valid": valid,
        "case_count": len(case_ids),
        "npz_count": len(files),
        "metadata_case_count": len(metadata_files),
        "expected_slices_from_metadata": expected_slices,
        "zero_byte_count": len(zero_byte),
        "invalid_zip_count": len(invalid_zip),
        "missing_member_count": len(missing_members),
        "metadata_error_count": len(metadata_errors),
        "layout_error_count": len(layout_errors),
        "required_processed_format": required_format,
        "examples": {
            "zero_byte": zero_byte[:5],
            "invalid_zip": invalid_zip[:5],
            "missing_members": missing_members[:5],
            "metadata_errors": metadata_errors[:5],
            "layout_errors": layout_errors[:5],
        },
    }
    return valid, report


def restore_cache(cache_path: Path, work_root: Path, expected_cases: int = 147) -> bool:
    if not cache_path.exists():
        return False
    processed_root = work_root / "processed"
    if processed_root.exists():
        shutil.rmtree(processed_root)
    print(f"Restoring processed cache from {cache_path}")
    try:
        with tarfile.open(cache_path, "r:*") as archive:
            try:
                archive.extractall(work_root, filter="data")
            except TypeError:
                archive.extractall(work_root)
    except (tarfile.TarError, EOFError, OSError) as exc:
        print(f"Processed cache extraction failed: {exc}")
        if processed_root.exists():
            shutil.rmtree(processed_root)
        return False
    valid, report = validate_processed_dataset(processed_root, expected_cases=expected_cases)
    print("Processed cache validation:")
    print(json.dumps(report, indent=2))
    if not valid:
        print("Ignoring invalid/incomplete processed cache and rebuilding from the source archive.")
        if processed_root.exists():
            shutil.rmtree(processed_root)
        invalid_name = cache_path.with_name(cache_path.name + ".invalid")
        try:
            if invalid_name.exists():
                invalid_name.unlink()
            cache_path.replace(invalid_name)
            print(f"Moved invalid cache to: {invalid_name}")
        except OSError:
            pass
        return False
    return True


def create_cache(processed_root: Path, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    print(f"Creating reusable processed cache: {cache_path}")
    with tarfile.open(temporary, "w") as archive:
        archive.add(processed_root, arcname="processed")
    temporary.replace(cache_path)


def backup_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end v1a.1 dual-kidney Colab pipeline")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--project-dir", default="/content/drive/MyDrive/medsam-kits23")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit MedSAM checkpoint; otherwise auto-discover in MyDrive root")
    parser.add_argument("--work-root", default="/content/v1a1_work")
    parser.add_argument("--split-csv", default=str(ROOT / "splits" / "v05_case_split.csv"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument("--run-medsam", action="store_true")
    parser.add_argument("--medsam-max-samples", type=int, default=100)
    parser.add_argument("--checkpoint-kind", choices=["best_continuous", "best_prompt"], default="best_continuous")
    parser.add_argument(
        "--stop-after",
        choices=["data", "manifest", "sanity", "overfit", "train", "predict", "complete"],
        default="complete",
    )
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    project_dir = Path(args.project_dir)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    assets = discover_assets(drive_root, project_dir, args.checkpoint)
    archive_path = Path(assets["data_archive"])
    checkpoint_path = Path(assets["medsam_checkpoint"])
    cache_path = Path(assets["processed_cache"])
    drive_results = Path(assets["results_dir"])
    drive_results.mkdir(parents=True, exist_ok=True)
    (drive_results / "assets.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")

    raw_root = work_root / "raw"
    processed_root = work_root / "processed"
    audit_dir = work_root / "orientation"
    manifest_dir = work_root / "manifest"
    overfit_dir = work_root / "overfit"
    run_dir = work_root / "run"

    processed_ready, local_report = validate_processed_dataset(processed_root, expected_cases=147)
    if processed_root.exists():
        print("Local processed-data validation:")
        print(json.dumps(local_report, indent=2))
    if args.rebuild_processed:
        if processed_root.exists():
            shutil.rmtree(processed_root)
        processed_ready = False
    if not processed_ready and not args.rebuild_processed:
        processed_ready = restore_cache(cache_path, work_root, expected_cases=147)
    if not processed_ready:
        run(py("sync_archive.py", "--source", str(archive_path), "--destination", str(raw_root)))
        run(py("audit_orientation.py", "--raw-root", str(raw_root), "--output-dir", str(audit_dir)))
        run(py("prepare_kits23_2d.py", "--raw-root", str(raw_root), "--output-root", str(processed_root)))
        processed_ready, rebuilt_report = validate_processed_dataset(processed_root, expected_cases=147)
        print("Rebuilt processed-data validation:")
        print(json.dumps(rebuilt_report, indent=2))
        if not processed_ready:
            raise RuntimeError("Rebuilt processed dataset failed integrity/completeness validation")
        create_cache(processed_root, cache_path)
        backup_tree(audit_dir, drive_results / "orientation")
    print(f"Processed NPZ count: {sum(1 for _ in processed_root.rglob('*.npz'))}")
    if args.stop_after == "data":
        return

    run(py(
        "build_dual_manifest.py",
        "--processed-root", str(processed_root),
        "--output-dir", str(manifest_dir),
        "--existing-split-csv", args.split_csv,
        "--target-labels", "1",
        "--min-component-area", "5",
        "--max-track-gap", "3",
    ))
    backup_tree(manifest_dir, drive_results / "manifest")
    manifest = manifest_dir / "v1a1_dual_manifest.csv"
    if args.stop_after == "manifest":
        return

    run(py(
        "sanity_check_dual.py",
        "--manifest", str(manifest),
        "--output", str(manifest_dir / "dual_target_sanity.png"),
        "--split", "train",
        "--num-samples", "10",
    ))
    shutil.copy2(manifest_dir / "dual_target_sanity.png", drive_results / "manifest" / "dual_target_sanity.png")
    if args.stop_after == "sanity":
        return

    if not args.skip_overfit:
        run(py(
            "overfit_dual.py",
            "--manifest", str(manifest),
            "--output-dir", str(overfit_dir),
            "--steps", "1200",
            "--batch-size", "12",
            "--base-channels", str(args.base_channels),
            "--device", "cuda",
        ))
        backup_tree(overfit_dir, drive_results / "overfit")
    if args.stop_after == "overfit":
        return

    run(py(
        "train_dual.py",
        "--manifest", str(manifest),
        "--output-dir", str(run_dir),
        "--backup-dir", str(drive_results / "run"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--base-channels", str(args.base_channels),
        "--lr", "3e-4",
        "--warmup-epochs", "3",
        "--metric-expansion-scale", "1.15",
        "--early-stopping-patience", "10",
        "--device", "cuda",
    ))
    if args.stop_after == "train":
        return

    selected_checkpoint = run_dir / f"{args.checkpoint_kind}.pt"
    for split in ("val", "test"):
        prediction_path = run_dir / f"{split}_predictions.csv"
        run(py(
            "predict_dual.py",
            "--manifest", str(manifest),
            "--checkpoint", str(selected_checkpoint),
            "--output-csv", str(prediction_path),
            "--split", split,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", "cuda",
        ))
        run(py(
            "evaluate_dual_prompts.py",
            "--predictions", str(prediction_path),
            "--output-dir", str(run_dir / f"{split}_prompt_evaluation_1p15"),
            "--variant", "both",
            "--expansion-scale", "1.15",
        ))
    run(py(
        "sweep_expansion.py",
        "--predictions", str(run_dir / "val_predictions.csv"),
        "--output-dir", str(run_dir / "val_expansion_sweep"),
        "--scales", "1.10,1.15,1.20,1.25",
    ))
    backup_tree(run_dir, drive_results / "run")
    if args.stop_after == "predict":
        return

    if args.run_medsam:
        medsam_repo = work_root / "MedSAM"
        local_checkpoint = work_root / "medsam_weights" / "medsam_vit_b.pth"
        run(py(
            "setup_medsam.py",
            "--repo-dir", str(medsam_repo),
            "--checkpoint-source", str(checkpoint_path),
            "--checkpoint-dest", str(local_checkpoint),
        ))
        run(py(
            "evaluate_medsam_dual.py",
            "--predictions", str(run_dir / "test_predictions.csv"),
            "--medsam-repo", str(medsam_repo),
            "--checkpoint", str(local_checkpoint),
            "--output-dir", str(run_dir / "medsam_test_pilot"),
            "--expansion-scale", "1.15",
            "--max-samples", str(args.medsam_max_samples),
            "--device", "cuda",
        ))
        backup_tree(run_dir, drive_results / "run")

    print(f"v1a.1 pipeline completed. Drive results: {drive_results}")


if __name__ == "__main__":
    main()
