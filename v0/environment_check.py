from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PATHS = {
    "medsam_root": Path("/content/MedSAM"),
    "checkpoint": Path(
        "/content/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
    ),
    "kits_dataset": Path("/content/kits23/dataset"),
    "project_root": Path(
        "/content/language_prompt_medsam_starter"
    ),
    "processed_root": Path("/content/kits23_2d_all"),
    "random_result": Path(
        "/content/v0_kidney_random_100"
    ),
    "one_per_case_result": Path(
        "/content/v0_kidney_one_per_case"
    ),
    "selected_case_list": Path(
        "/content/kits23_selected_100_cases.json"
    ),
}

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "nibabel": "nibabel",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "torch": "torch",
    "segment_anything": "segment_anything",
}

TARGET_NAMES = {
    0: "background",
    1: "kidney",
    2: "tumor",
    3: "cyst",
}


def human_size(number_of_bytes: int | float) -> str:
    value = float(number_of_bytes)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} TiB"


def file_md5(
    path: Path,
    chunk_size: int = 4 * 1024 * 1024,
) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0

    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            pass

    return total


def check_packages() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for display_name, module_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)

            version = getattr(
                module,
                "__version__",
                "installed",
            )

            results[display_name] = {
                "ok": True,
                "version": str(version),
            }

        except Exception as error:
            results[display_name] = {
                "ok": False,
                "error": (
                    f"{type(error).__name__}: {error}"
                ),
            }

    return results


def check_cuda() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_imported": False,
        "cuda_available": False,
    }

    try:
        import torch

        result["torch_imported"] = True
        result["torch_version"] = torch.__version__
        result["cuda_available"] = (
            torch.cuda.is_available()
        )

        if torch.cuda.is_available():
            result["cuda_device_count"] = (
                torch.cuda.device_count()
            )
            result["cuda_device_name"] = (
                torch.cuda.get_device_name(0)
            )
            result["cuda_version"] = (
                torch.version.cuda
            )

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def check_git(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "git_available": False,
        "is_repository": False,
    }

    try:
        git_version = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        result["git_available"] = True
        result["git_version"] = (
            git_version.stdout.strip()
        )

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )
        return result

    if not path.exists():
        return result

    repository_test = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "--is-inside-work-tree",
        ],
        capture_output=True,
        text=True,
    )

    result["is_repository"] = (
        repository_test.returncode == 0
        and repository_test.stdout.strip() == "true"
    )

    if result["is_repository"]:
        branch = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "branch",
                "--show-current",
            ],
            capture_output=True,
            text=True,
        )
        result["branch"] = branch.stdout.strip()

        remote = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "remote",
                "-v",
            ],
            capture_output=True,
            text=True,
        )
        result["remotes"] = [
            line
            for line in remote.stdout.splitlines()
            if line.strip()
        ]

        status = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--short",
            ],
            capture_output=True,
            text=True,
        )
        result["uncommitted_changes"] = [
            line
            for line in status.stdout.splitlines()
            if line.strip()
        ]

    return result


def check_project_scripts(
    project_root: Path,
) -> dict[str, Any]:
    expected_scripts = [
        "prepare_kits23_2d.py",
        "run_v0_experiment.py",
    ]

    result: dict[str, Any] = {
        "path": str(project_root),
        "exists": project_root.exists(),
        "scripts": {},
    }

    for script_name in expected_scripts:
        script_path = project_root / script_name

        script_result: dict[str, Any] = {
            "path": str(script_path),
            "exists": script_path.exists(),
        }

        if script_path.exists():
            script_result["size"] = (
                script_path.stat().st_size
            )
            script_result["size_human"] = human_size(
                script_path.stat().st_size
            )

            compilation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(script_path),
                ],
                capture_output=True,
                text=True,
            )

            script_result["syntax_ok"] = (
                compilation.returncode == 0
            )

            if compilation.returncode != 0:
                script_result["syntax_error"] = (
                    compilation.stderr.strip()
                )

        result["scripts"][script_name] = script_result

    return result


def check_checkpoint(
    checkpoint_path: Path,
    calculate_md5: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
    }

    if not checkpoint_path.exists():
        return result

    size = checkpoint_path.stat().st_size
    result["size"] = size
    result["size_human"] = human_size(size)
    result["looks_large_enough"] = (
        size > 300 * 1024 * 1024
    )

    if calculate_md5:
        result["md5"] = file_md5(checkpoint_path)

    return result


def discover_complete_cases(
    dataset_root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    complete_cases: list[Path] = []
    incomplete_cases: list[dict[str, Any]] = []

    if not dataset_root.exists():
        return complete_cases, incomplete_cases

    for case_directory in sorted(
        dataset_root.glob("case_*")
    ):
        image_path = (
            case_directory / "imaging.nii.gz"
        )
        segmentation_path = (
            case_directory / "segmentation.nii.gz"
        )

        if image_path.exists() and segmentation_path.exists():
            complete_cases.append(case_directory)
        else:
            incomplete_cases.append(
                {
                    "case_id": case_directory.name,
                    "has_imaging": image_path.exists(),
                    "has_segmentation": (
                        segmentation_path.exists()
                    ),
                }
            )

    return complete_cases, incomplete_cases


def inspect_sample_case(
    case_directory: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case_directory.name,
    }

    try:
        import nibabel as nib

        image_path = (
            case_directory / "imaging.nii.gz"
        )
        segmentation_path = (
            case_directory / "segmentation.nii.gz"
        )

        image_nii = nib.load(str(image_path))
        segmentation_nii = nib.load(
            str(segmentation_path)
        )

        image_shape = tuple(
            int(value) for value in image_nii.shape
        )
        segmentation_shape = tuple(
            int(value)
            for value in segmentation_nii.shape
        )

        segmentation = np.rint(
            segmentation_nii.get_fdata(
                dtype=np.float32
            )
        ).astype(np.uint8)

        unique_labels, counts = np.unique(
            segmentation,
            return_counts=True,
        )

        result.update(
            {
                "image_shape": image_shape,
                "segmentation_shape": (
                    segmentation_shape
                ),
                "shapes_match": (
                    image_shape == segmentation_shape
                ),
                "image_dtype": str(
                    image_nii.get_data_dtype()
                ),
                "segmentation_dtype": str(
                    segmentation_nii.get_data_dtype()
                ),
                "voxel_spacing": [
                    float(value)
                    for value in (
                        image_nii.header.get_zooms()[:3]
                    )
                ],
                "labels": {
                    str(int(label)): {
                        "name": TARGET_NAMES.get(
                            int(label),
                            "unknown",
                        ),
                        "voxel_count": int(count),
                    }
                    for label, count in zip(
                        unique_labels,
                        counts,
                    )
                },
            }
        )

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def check_dataset(
    dataset_root: Path,
) -> dict[str, Any]:
    complete_cases, incomplete_cases = (
        discover_complete_cases(dataset_root)
    )

    result: dict[str, Any] = {
        "path": str(dataset_root),
        "exists": dataset_root.exists(),
        "complete_case_count": len(complete_cases),
        "incomplete_case_count": len(
            incomplete_cases
        ),
        "incomplete_cases": incomplete_cases[:20],
    }

    if complete_cases:
        result["first_complete_case"] = (
            complete_cases[0].name
        )
        result["sample_case"] = inspect_sample_case(
            complete_cases[0]
        )

    return result


def check_processed_data(
    processed_root: Path,
    full_npz_scan: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(processed_root),
        "exists": processed_root.exists(),
    }

    index_path = processed_root / "index.csv"
    slices_directory = processed_root / "slices"

    result["index_exists"] = index_path.exists()
    result["slices_directory_exists"] = (
        slices_directory.exists()
    )

    if not index_path.exists():
        return result

    try:
        import pandas as pd

        index = pd.read_csv(index_path)

        result["index_rows"] = len(index)
        result["columns"] = index.columns.tolist()

        if "case_id" in index.columns:
            result["indexed_case_count"] = int(
                index["case_id"].nunique()
            )

        if "target_name" in index.columns:
            result["rows_by_target"] = {
                str(key): int(value)
                for key, value in (
                    index.groupby("target_name")
                    .size()
                    .to_dict()
                    .items()
                )
            }

        if "npz_path" in index.columns:
            paths_to_check = index["npz_path"].drop_duplicates()

            if not full_npz_scan:
                paths_to_check = paths_to_check.head(100)

            missing_paths = []

            for relative_path in paths_to_check:
                absolute_path = (
                    processed_root / str(relative_path)
                )

                if not absolute_path.exists():
                    missing_paths.append(
                        str(relative_path)
                    )

            result["npz_paths_checked"] = len(
                paths_to_check
            )
            result["missing_npz_count"] = len(
                missing_paths
            )
            result["missing_npz_examples"] = (
                missing_paths[:20]
            )
            result["full_npz_scan"] = full_npz_scan

            first_relative_path = (
                index["npz_path"].iloc[0]
            )
            first_npz_path = (
                processed_root
                / str(first_relative_path)
            )

            if first_npz_path.exists():
                with np.load(first_npz_path) as sample:
                    result["sample_npz"] = {
                        "path": str(
                            first_npz_path
                        ),
                        "keys": list(sample.keys()),
                        "image_shape": list(
                            sample["image"].shape
                        ),
                        "image_dtype": str(
                            sample["image"].dtype
                        ),
                        "image_min": float(
                            sample["image"].min()
                        ),
                        "image_max": float(
                            sample["image"].max()
                        ),
                        "label_shape": list(
                            sample["label"].shape
                        ),
                        "label_dtype": str(
                            sample["label"].dtype
                        ),
                        "label_values": [
                            int(value)
                            for value in np.unique(
                                sample["label"]
                            )
                        ],
                    }

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def check_result_directory(
    result_directory: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(result_directory),
        "exists": result_directory.exists(),
    }

    if not result_directory.exists():
        return result

    expected_files = [
        "selected_samples.csv",
        "per_sample_results.csv",
        "summary.csv",
    ]

    result["files"] = {
        file_name: (
            result_directory / file_name
        ).exists()
        for file_name in expected_files
    }

    per_sample_path = (
        result_directory / "per_sample_results.csv"
    )

    if per_sample_path.exists():
        try:
            import pandas as pd

            data_frame = pd.read_csv(
                per_sample_path
            )

            result["result_rows"] = len(data_frame)

            if "case_id" in data_frame.columns:
                result["evaluated_cases"] = int(
                    data_frame["case_id"].nunique()
                )

            if {
                "case_id",
                "slice_index",
                "target_name",
            }.issubset(data_frame.columns):
                result["evaluated_samples"] = int(
                    len(
                        data_frame[
                            [
                                "case_id",
                                "slice_index",
                                "target_name",
                            ]
                        ].drop_duplicates()
                    )
                )

            if {
                "mode",
                "dice",
            }.issubset(data_frame.columns):
                mean_dice = (
                    data_frame.groupby("mode")["dice"]
                    .mean()
                    .sort_values(ascending=False)
                )

                result["mean_dice_by_mode"] = {
                    str(key): float(value)
                    for key, value in (
                        mean_dice.to_dict().items()
                    )
                }

        except Exception as error:
            result["error"] = (
                f"{type(error).__name__}: {error}"
            )

    visualization_directory = (
        result_directory / "visualizations"
    )

    if visualization_directory.exists():
        result["visualization_count"] = len(
            list(
                visualization_directory.glob(
                    "*.png"
                )
            )
        )

    return result


def check_selected_case_list(
    path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return result

    try:
        content = json.loads(
            path.read_text(encoding="utf-8")
        )

        result["keys"] = list(content.keys())

        case_ids = content.get("case_ids", [])
        result["case_id_count"] = len(case_ids)
        result["first_case_ids"] = case_ids[:10]
        result["random_seed"] = content.get(
            "random_seed"
        )

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def collect_report(
    paths: dict[str, Path],
    calculate_md5: bool,
    full_npz_scan: bool,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "system": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "working_directory": os.getcwd(),
        },
        "packages": check_packages(),
        "cuda": check_cuda(),
        "git": check_git(paths["project_root"]),
        "project_scripts": check_project_scripts(
            paths["project_root"]
        ),
        "checkpoint": check_checkpoint(
            paths["checkpoint"],
            calculate_md5=calculate_md5,
        ),
        "dataset": check_dataset(
            paths["kits_dataset"]
        ),
        "processed_data": check_processed_data(
            paths["processed_root"],
            full_npz_scan=full_npz_scan,
        ),
        "selected_case_list": (
            check_selected_case_list(
                paths["selected_case_list"]
            )
        ),
        "results": {
            "random": check_result_directory(
                paths["random_result"]
            ),
            "one_per_case": check_result_directory(
                paths["one_per_case_result"]
            ),
        },
        "directory_sizes": {
            name: {
                "path": str(path),
                "size_bytes": directory_size(path),
                "size_human": human_size(
                    directory_size(path)
                ),
            }
            for name, path in {
                "project_root": paths["project_root"],
                "processed_root": paths["processed_root"],
                "random_result": paths["random_result"],
                "one_per_case_result": (
                    paths["one_per_case_result"]
                ),
            }.items()
        },
    }


def print_status(
    label: str,
    passed: bool,
    details: str = "",
) -> None:
    status = "[OK]" if passed else "[MISSING]"
    suffix = f" — {details}" if details else ""
    print(f"{status:<9} {label}{suffix}")


def print_summary(report: dict[str, Any]) -> None:
    print("\n========== MedSAM 项目最终环境检查 ==========\n")

    cuda = report["cuda"]
    print_status(
        "CUDA",
        bool(cuda.get("cuda_available")),
        str(cuda.get("cuda_device_name", "unavailable")),
    )

    checkpoint = report["checkpoint"]
    print_status(
        "MedSAM checkpoint",
        bool(checkpoint.get("exists")),
        str(checkpoint.get("size_human", "")),
    )

    scripts = report["project_scripts"]["scripts"]

    for script_name, script_result in scripts.items():
        print_status(
            script_name,
            bool(
                script_result.get("exists")
                and script_result.get("syntax_ok")
            ),
            (
                "syntax OK"
                if script_result.get("syntax_ok")
                else ""
            ),
        )

    dataset = report["dataset"]
    print_status(
        "KiTS23 complete cases",
        dataset.get("complete_case_count", 0) > 0,
        str(dataset.get("complete_case_count", 0)),
    )

    processed = report["processed_data"]
    print_status(
        "Processed index.csv",
        bool(processed.get("index_exists")),
        (
            f"{processed.get('indexed_case_count', 0)} cases, "
            f"{processed.get('index_rows', 0)} rows"
        ),
    )

    print_status(
        "Processed NPZ references",
        processed.get("missing_npz_count", 1) == 0,
        (
            f"checked {processed.get('npz_paths_checked', 0)}, "
            f"missing {processed.get('missing_npz_count', 0)}"
        ),
    )

    selected_list = report["selected_case_list"]
    print_status(
        "Fixed selected-case JSON",
        bool(selected_list.get("exists")),
        (
            f"{selected_list.get('case_id_count', 0)} cases"
            if selected_list.get("exists")
            else "optional but recommended"
        ),
    )

    for result_name, result in report["results"].items():
        print_status(
            f"Result: {result_name}",
            bool(result.get("exists")),
            (
                f"{result.get('evaluated_cases', 0)} cases"
                if result.get("exists")
                else ""
            ),
        )

        if result.get("mean_dice_by_mode"):
            print("           Mean Dice:")
            for mode, score in (
                result["mean_dice_by_mode"].items()
            ):
                print(
                    f"             {mode:<14} {score:.6f}"
                )

    git = report["git"]
    print_status(
        "Git repository",
        bool(git.get("is_repository")),
        (
            git.get("branch", "")
            if git.get("is_repository")
            else "not initialized"
        ),
    )

    print("\nDirectory sizes:")

    for name, information in (
        report["directory_sizes"].items()
    ):
        print(
            f"  {name:<22} "
            f"{information['size_human']}"
        )

    print("\n=============================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the current MedSAM + KiTS23 Colab "
            "environment and save a reproducibility report."
        )
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "/content/medsam_environment_report.json"
        ),
        help="Path used to save the JSON report.",
    )
    parser.add_argument(
        "--md5",
        action="store_true",
        help=(
            "Calculate the MedSAM checkpoint MD5. "
            "This takes additional time."
        ),
    )
    parser.add_argument(
        "--full-npz-scan",
        action="store_true",
        help=(
            "Check every distinct NPZ reference in index.csv. "
            "The default checks the first 100."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = collect_report(
        paths=DEFAULT_PATHS,
        calculate_md5=args.md5,
        full_npz_scan=args.full_npz_scan,
    )

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_summary(report)
    print("JSON report saved to:")
    print(args.report)


if __name__ == "__main__":
    main()
