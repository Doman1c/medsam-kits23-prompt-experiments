from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.data import DualKidneyDataset, load_npz
from v1a1.losses import compute_dual_loss
from v1a1.model import DualKidneyPromptNet
from v1a1.runtime import evaluate_model, move_batch_to_device
from v1a1.utils import resolve_device, save_json, set_seed


def select_subset(frame: pd.DataFrame, positives: int, negatives: int, seed: int) -> pd.DataFrame:
    train = frame[frame.split == "train"].copy()
    selected = []
    categories = [
        train[(train.left_presence == 1) & (train.right_presence == 1)],
        train[(train.left_presence == 1) & (train.right_presence == 0)],
        train[(train.left_presence == 0) & (train.right_presence == 1)],
        train[(train.left_area_bin == "00_10") | (train.right_area_bin == "00_10")],
        train[(train.left_area_bin == "10_50") | (train.right_area_bin == "10_50")],
        train[(train.left_area_bin == "90_100") | (train.right_area_bin == "90_100")],
    ]
    per_category = max(1, positives // len(categories))
    used = set()
    for index, category in enumerate(categories):
        category = category[~category.index.isin(used)]
        if len(category):
            sample = category.sample(min(per_category, len(category)), random_state=seed + index)
            selected.append(sample)
            used.update(sample.index.tolist())
    positive_frame = pd.concat(selected) if selected else train.iloc[:0]
    if len(positive_frame) < positives:
        pool = train[(train.has_any == 1) & (~train.index.isin(used))]
        if len(pool):
            extra = pool.sample(min(positives - len(positive_frame), len(pool)), random_state=seed + 50)
            positive_frame = pd.concat([positive_frame, extra])
    negative_pool = train[train.has_any == 0]
    negative_frame = negative_pool.sample(min(negatives, len(negative_pool)), random_state=seed + 100)
    subset = pd.concat([positive_frame.head(positives), negative_frame]).sort_values(["case_id", "slice_idx"])
    return subset


def save_prediction_plot(frame: pd.DataFrame, output: Path, count: int = 8) -> None:
    sample = frame[frame.left_gt_presence + frame.right_gt_presence > 0].head(count)
    figure, axes = plt.subplots(len(sample), 1, figsize=(7, 5 * max(len(sample), 1)))
    axes = np.atleast_1d(axes)
    for axis, (_, row) in zip(axes, sample.iterrows()):
        image, _ = load_npz(row.npz_path)
        axis.imshow(image, cmap="gray")
        for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
            if int(row[f"{side}_gt_presence"]):
                box = [row[f"{side}_gt_{key}"] for key in ("x1", "y1", "x2", "y2")]
                axis.add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=color, linewidth=2))
            if int(row[f"{side}_raw_presence"]):
                box = [row[f"{side}_raw_{key}"] for key in ("x1", "y1", "x2", "y2")]
                axis.add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor="red", linewidth=1, linestyle="--"))
        axis.set_title(f"{row.case_id} z={int(row.slice_idx)}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic dual-kidney overfit gate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-slices", type=int, default=48)
    parser.add_argument("--negative-slices", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--expansion-scale", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest)
    subset = select_subset(frame, args.positive_slices, args.negative_slices, args.seed)
    subset_path = output_dir / "overfit_subset.csv"
    subset.to_csv(subset_path, index=False)

    dataset = DualKidneyDataset(
        subset_path,
        "train",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        input_mode="2.5d",
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    iterator = itertools.cycle(loader)

    model = DualKidneyPromptNet(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    best_summary = None
    best_frame = None
    best_score = -float("inf")
    history = []

    for step in trange(1, args.steps + 1, desc="Dual overfit check"):
        model.train()
        batch = move_batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["image"])
        loss, _ = compute_dual_loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            summary, prediction_frame = evaluate_model(
                model,
                eval_loader,
                device,
                expansion_scale=args.expansion_scale,
                presence_threshold=0.5,
                use_amp=False,
            )
            summary["step"] = step
            history.append(summary)
            score = summary["val_continuous_score"] + 0.5 * summary["val_prompt_success_rate"]
            if score > best_score:
                best_score = score
                best_summary = dict(summary)
                best_frame = prediction_frame.copy()
                torch.save(
                    {"step": step, "model_state": model.state_dict(), "summary": summary, "config": vars(args)},
                    output_dir / "best_overfit.pt",
                )

    assert best_summary is not None and best_frame is not None
    passed = (
        best_summary["val_positive_recall"] >= 0.98
        and best_summary["val_center_error_pixels_median"] <= 6.0
        and best_summary["val_expanded_coverage_mean"] >= 0.97
        and best_summary["val_prompt_success_rate"] >= 0.85
        and best_summary["val_side_order_violation_rate"] == 0.0
        and best_summary["val_tight_box_iou"] >= 0.75
    )
    result = {
        "passed": bool(passed),
        "criteria": {
            "positive_recall": ">=0.98",
            "center_error_pixels_median": "<=6",
            "expanded_coverage_mean": ">=0.97",
            "prompt_success_rate": ">=0.85",
            "side_order_violation_rate": "==0",
            "tight_box_iou": ">=0.75",
        },
        "best_metrics": best_summary,
    }
    save_json(output_dir / "overfit_history.json", history)
    save_json(output_dir / "overfit_result.json", result)
    best_frame.to_csv(output_dir / "overfit_predictions.csv", index=False)
    save_prediction_plot(best_frame, output_dir / "overfit_predictions.png")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("DUAL OVERFIT CHECK FAILED: inspect targets and predictions before full training")
    print("DUAL OVERFIT CHECK PASSED")


if __name__ == "__main__":
    main()
