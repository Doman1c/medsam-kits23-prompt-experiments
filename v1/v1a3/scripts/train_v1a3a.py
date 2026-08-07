from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THIS_ROOT))
sys.path.insert(0, str(V1_ROOT / "v1a2b"))

from v1a2b.data import DualKidneyDataset
from v1a2b.ema import ModelEMA
from v1a2b.losses import compute_dual_loss
from v1a2b.model import DualKidneyPromptNet, count_parameters
from v1a2b.runtime import evaluate_model, move_batch_to_device
from v1a2b.utils import resolve_device, save_json, set_seed

from v1a3.sampling import SamplerConfig, build_presence_repair_sampler
from v1a3.selection import (
    geometry_checkpoint_score,
    joint_checkpoint_score,
    presence_checkpoint_score,
)


def copy_training_state(source_dir: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "history.json",
        "config.json",
        "sampler_audit.json",
        "checkpoint_summary.json",
        "best_presence.pt",
        "best_geometry.pt",
        "best_joint.pt",
        "last.pt",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train V1a.3-A. Architecture and optimized loss are identical "
            "to V1a.2-B; targeted presence sampling is the intended change."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--hard-negative-csv", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument(
        "--input-mode",
        choices=["2d", "2.5d"],
        default="2.5d",
    )
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--early-stopping-patience", type=int, default=12)

    parser.add_argument("--negative-fraction", type=float, default=0.45)
    parser.add_argument("--mined-negative-share", type=float, default=0.45)
    parser.add_argument("--boundary-negative-share", type=float, default=0.35)
    parser.add_argument("--near-negative-share", type=float, default=0.15)
    parser.add_argument("--far-negative-share", type=float, default=0.05)
    parser.add_argument("--boundary-radius", type=int, default=8)
    parser.add_argument("--hard-negative-min-score", type=float, default=0.20)
    parser.add_argument("--hard-negative-top-k-per-case", type=int, default=32)
    parser.add_argument("--unilateral-positive-boost", type=float, default=1.50)

    parser.add_argument("--metric-expansion-scale", type=float, default=1.15)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--minimum-left-recall", type=float, default=0.90)
    parser.add_argument("--minimum-right-recall", type=float, default=0.90)

    parser.add_argument("--pred-center-size-weight", type=float, default=0.5)
    parser.add_argument("--size-consistency-weight", type=float, default=0.0)
    parser.add_argument("--decoded-box-weight", type=float, default=0.5)
    parser.add_argument("--softargmax-beta", type=float, default=12.0)
    parser.add_argument(
        "--detach-pred-center",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not 0.0 <= args.pred_center_size_weight <= 1.0:
        parser.error("--pred-center-size-weight must be in [0,1]")
    if args.decoded_box_weight < 0:
        parser.error("--decoded-box-weight must be non-negative")

    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(args.backup_dir) if args.backup_dir else None

    train_dataset = DualKidneyDataset(
        args.manifest,
        "train",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        input_mode=args.input_mode,
        augment=True,
    )
    val_dataset = DualKidneyDataset(
        args.manifest,
        "val",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        input_mode=args.input_mode,
        augment=False,
    )

    sampler_config = SamplerConfig(
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
    sampler, sampler_audit, sampler_frame = build_presence_repair_sampler(
        train_dataset,
        config=sampler_config,
        hard_negative_csv=args.hard_negative_csv,
    )
    save_json(output_dir / "sampler_audit.json", sampler_audit)
    sampler_frame.to_csv(
        output_dir / "sampler_row_classification.csv",
        index=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = DualKidneyPromptNet(
        in_channels=1 if args.input_mode == "2d" else 3,
        base_channels=args.base_channels,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < args.warmup_epochs:
            return float(epoch_index + 1) / max(args.warmup_epochs, 1)
        decay_epochs = max(args.epochs - args.warmup_epochs, 1)
        progress = (
            epoch_index - args.warmup_epochs
        ) / decay_epochs
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    loss_kwargs = {
        "pred_center_size_weight": args.pred_center_size_weight,
        "size_consistency_weight": args.size_consistency_weight,
        "decoded_box_weight": args.decoded_box_weight,
        "softargmax_beta": args.softargmax_beta,
        "detach_pred_center": args.detach_pred_center,
    }
    config = vars(args) | {
        "experiment": "V1a.3-A",
        "base_geometry": "V1a.2-B unchanged",
        "controlled_change": (
            "hard-negative, boundary-negative, and unilateral-positive "
            "sampling; no architecture or optimized-loss change"
        ),
        "loss_kwargs": loss_kwargs,
        "sampler": sampler_audit,
        "device_resolved": str(device),
        "trainable_parameters": count_parameters(model),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
    }
    save_json(output_dir / "config.json", config)
    print(json.dumps(config, indent=2))

    history: list[dict] = []
    best_scores = {
        "presence": -float("inf"),
        "geometry": -float("inf"),
        "joint": -float("inf"),
    }
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        progress = tqdm(
            train_loader,
            desc=f"V1a.3-A epoch {epoch:02d}/{args.epochs:02d}",
        )

        for batch_cpu in progress:
            batch = move_batch_to_device(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ):
                outputs = model(batch["image"])
                loss, metrics = compute_dual_loss(
                    outputs,
                    batch,
                    **loss_kwargs,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}: {metrics}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            for key, value in metrics.items():
                sums[key] += float(value)
            batches += 1
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                presence=f"{metrics['presence_loss']:.4f}",
                box=f"{metrics['decoded_box_loss']:.4f}",
            )

        train_metrics = {
            f"train_{key}": value / max(batches, 1)
            for key, value in sums.items()
        }
        val_metrics, val_predictions = evaluate_model(
            ema.module,
            val_loader,
            device,
            expansion_scale=args.metric_expansion_scale,
            presence_threshold=args.presence_threshold,
            loss_kwargs=loss_kwargs,
        )
        scheduler.step()

        presence_score, presence_details = presence_checkpoint_score(
            val_metrics,
            min_recall=args.minimum_recall,
            min_left_recall=args.minimum_left_recall,
            min_right_recall=args.minimum_right_recall,
        )
        geometry_score = geometry_checkpoint_score(val_metrics)
        joint_score, joint_details = joint_checkpoint_score(
            val_metrics,
            min_recall=args.minimum_recall,
            min_left_recall=args.minimum_left_recall,
            min_right_recall=args.minimum_right_recall,
        )

        row = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "presence_checkpoint_score": presence_score,
            "geometry_checkpoint_score": geometry_score,
            "joint_checkpoint_score": joint_score,
            "presence_checkpoint_feasible": presence_details["feasible"],
            **train_metrics,
            **val_metrics,
        }
        history.append(row)
        save_json(output_dir / "history.json", history)

        checkpoint = {
            "epoch": epoch,
            "experiment": "V1a.3-A",
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "metrics": row,
            "presence_selection": presence_details,
            "joint_selection": joint_details,
        }
        epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, output_dir / "last.pt")

        improved = False
        for name, score in {
            "presence": presence_score,
            "geometry": geometry_score,
            "joint": joint_score,
        }.items():
            if score > best_scores[name]:
                best_scores[name] = score
                torch.save(
                    checkpoint,
                    output_dir / f"best_{name}.pt",
                )
                val_predictions.to_csv(
                    output_dir / f"best_{name}_val_raw_predictions.csv",
                    index=False,
                )
                improved = True

        epochs_without_improvement = (
            0 if improved else epochs_without_improvement + 1
        )
        print(json.dumps(row, indent=2))
        print(
            "Best presence={presence:.4f} | geometry={geometry:.4f} | "
            "joint={joint:.4f} | patience={patience}/{limit}".format(
                **best_scores,
                patience=epochs_without_improvement,
                limit=args.early_stopping_patience,
            )
        )

        if backup_dir:
            copy_training_state(output_dir, backup_dir)
            backup_checkpoint_dir = backup_dir / "checkpoints"
            backup_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                epoch_path,
                backup_checkpoint_dir / epoch_path.name,
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    save_json(
        output_dir / "checkpoint_summary.json",
        {
            "best_scores": best_scores,
            "epochs_completed": len(history),
            "selection_note": (
                "These are prompt-network proxies. Final model selection "
                "must use validation-only calibration and negative-aware MedSAM."
            ),
        },
    )
    if backup_dir:
        copy_training_state(output_dir, backup_dir)
    print(f"V1a.3-A training completed: {output_dir}")


if __name__ == "__main__":
    main()
