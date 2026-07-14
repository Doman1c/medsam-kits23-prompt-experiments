from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed case-level train/validation/test split from "
            "the KiTS23 2D index.csv file."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Directory containing index.csv and slices/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file, e.g. configs/splits_v1.json.",
    )
    parser.add_argument(
        "--target-name",
        choices=["kidney", "tumor", "cyst"],
        default="kidney",
        help="Only cases containing this indexed target are included.",
    )
    parser.add_argument("--train-cases", type=int, default=100)
    parser.add_argument("--val-cases", type=int, default=20)
    parser.add_argument("--test-cases", type=int, default=27)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.processed_root / "index.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found: {index_path}")

    data = pd.read_csv(index_path)
    required = {"case_id", "target_name"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"index.csv is missing required columns: {sorted(missing)}"
        )

    target_data = data[data["target_name"] == args.target_name].copy()
    case_ids = sorted(target_data["case_id"].astype(str).unique().tolist())

    requested_total = args.train_cases + args.val_cases + args.test_cases
    if requested_total != len(case_ids):
        raise ValueError(
            "Requested split sizes must exactly match the number of available "
            f"{args.target_name} cases.\n"
            f"Available cases: {len(case_ids)}\n"
            f"Requested: train={args.train_cases}, val={args.val_cases}, "
            f"test={args.test_cases}, total={requested_total}\n"
            "Adjust the three counts before creating the permanent split."
        )

    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(case_ids, dtype=object)
    rng.shuffle(shuffled)

    train_end = args.train_cases
    val_end = train_end + args.val_cases

    split = {
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(args.processed_root),
        "target_name": args.target_name,
        "seed": args.seed,
        "number_of_cases": len(case_ids),
        "train": sorted(shuffled[:train_end].tolist()),
        "val": sorted(shuffled[train_end:val_end].tolist()),
        "test": sorted(shuffled[val_end:].tolist()),
    }

    all_assigned = split["train"] + split["val"] + split["test"]
    if len(all_assigned) != len(set(all_assigned)):
        raise RuntimeError("The generated split contains duplicate case IDs.")
    if set(all_assigned) != set(case_ids):
        raise RuntimeError("The generated split does not cover every case.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(split, file, indent=2, ensure_ascii=False)

    print("Case split created.")
    print("Target:", args.target_name)
    print("Seed:", args.seed)
    print("Train cases:", len(split["train"]))
    print("Validation cases:", len(split["val"]))
    print("Test cases:", len(split["test"]))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
