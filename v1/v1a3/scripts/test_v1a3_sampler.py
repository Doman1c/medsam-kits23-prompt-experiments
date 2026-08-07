from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic sampler gate for V1a.3-A"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hard-negative-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--negative-fraction", type=float, default=0.45)
    parser.add_argument("--mined-negative-share", type=float, default=0.45)
    parser.add_argument("--boundary-negative-share", type=float, default=0.35)
    parser.add_argument("--near-negative-share", type=float, default=0.15)
    parser.add_argument("--far-negative-share", type=float, default=0.05)
    parser.add_argument("--boundary-radius", type=int, default=8)
    parser.add_argument("--hard-negative-min-score", type=float, default=0.20)
    parser.add_argument("--hard-negative-top-k-per-case", type=int, default=32)
    parser.add_argument("--unilateral-positive-boost", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root / "v1" / "v1a2b"))
    sys.path.insert(0, str(repo_root / "v1" / "v1a3"))

    from v1a2b.data import DualKidneyDataset
    from v1a3.sampling import SamplerConfig, build_presence_repair_sampler

    dataset = DualKidneyDataset(
        args.manifest,
        "train",
        input_size=256,
        heatmap_size=64,
        input_mode="2.5d",
        augment=False,
    )
    config = SamplerConfig(
        negative_fraction=args.negative_fraction,
        mined_negative_share=args.mined_negative_share,
        boundary_negative_share=args.boundary_negative_share,
        near_negative_share=args.near_negative_share,
        far_negative_share=args.far_negative_share,
        boundary_radius=args.boundary_radius,
        hard_negative_min_score=args.hard_negative_min_score,
        hard_negative_top_k_per_case=args.hard_negative_top_k_per_case,
        unilateral_positive_boost=args.unilateral_positive_boost,
        seed=args.seed,
    )
    sampler, audit, classified = build_presence_repair_sampler(
        dataset,
        config=config,
        hard_negative_csv=args.hard_negative_csv,
    )

    sampled_indices = list(iter(sampler))
    sampled_categories = classified.iloc[sampled_indices]["sampling_category"]
    counts = Counter(sampled_categories.tolist())
    fractions = {
        category: count / max(len(sampled_indices), 1)
        for category, count in sorted(counts.items())
    }
    realized_negative_fraction = sum(
        fraction
        for category, fraction in fractions.items()
        if category.startswith("negative_")
    )
    tolerance = max(0.04, 3.0 / np.sqrt(max(len(sampled_indices), 1)))
    passed = (
        abs(realized_negative_fraction - args.negative_fraction)
        <= tolerance
    )

    mined_available = audit["category_counts"].get("negative_mined", 0) > 0
    if args.hard_negative_csv and mined_available:
        passed = passed and fractions.get("negative_mined", 0.0) > 0.0

    payload = audit | {
        "sampled_rows": int(len(sampled_indices)),
        "realized_category_counts": dict(counts),
        "realized_category_fraction": fractions,
        "realized_negative_fraction": float(realized_negative_fraction),
        "negative_fraction_tolerance": float(tolerance),
        "passed": bool(passed),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(
        output_dir / "sampler_row_classification.csv",
        index=False,
    )
    (output_dir / "sampler_gate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("V1a.3-A sampler gate failed")
    print("V1a.3-A sampler gate passed")


if __name__ == "__main__":
    main()
