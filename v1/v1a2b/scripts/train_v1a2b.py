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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a2b.data import (
    DualKidneyDataset,
    build_dual_balanced_sampler,
)
from v1a2b.ema import ModelEMA
from v1a2b.losses import compute_dual_loss
from v1a2b.model import (
    DualKidneyPromptNet,
    count_parameters,
)
from v1a2b.runtime import (
    evaluate_model,
    move_batch_to_device,
)
from v1a2b.utils import (
    resolve_device,
    save_json,
    set_seed,
)


def copy_training_state(
    source_dir: Path,
    backup_dir: Path,
) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "history.json",
        "config.json",
        "best_prompt.pt",
        "best_continuous.pt",
        "last.pt",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copy2(
                source,
                backup_dir / name,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train V1a.2-B with predicted-center size "
            "and decoded-corner supervision"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--heatmap-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--input-mode",
        choices=["2d", "2.5d"],
        default="2.5d",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--near-negative-fraction",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--metric-expansion-scale",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--pred-center-size-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--size-consistency-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--decoded-box-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--softargmax-beta",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--detach-pred-center",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    args = parser.parse_args()

    if not 0.0 <= args.pred_center_size_weight <= 1.0:
        parser.error(
            "--pred-center-size-weight must be in [0,1]"
        )
    if args.decoded_box_weight < 0:
        parser.error(
            "--decoded-box-weight must be non-negative"
        )

    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    backup_dir = (
        Path(args.backup_dir)
        if args.backup_dir
        else None
    )

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

    sampler = build_dual_balanced_sampler(
        train_dataset,
        negative_ratio=args.negative_ratio,
        near_negative_fraction=(
            args.near_negative_fraction
        ),
        seed=args.seed,
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
        in_channels=(
            1 if args.input_mode == "2d" else 3
        ),
        base_channels=args.base_channels,
    ).to(device)
    ema = ModelEMA(
        model,
        decay=args.ema_decay,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < args.warmup_epochs:
            return float(
                epoch_index + 1
            ) / max(args.warmup_epochs, 1)

        decay_epochs = max(
            args.epochs - args.warmup_epochs,
            1,
        )
        progress = (
            epoch_index - args.warmup_epochs
        ) / decay_epochs
        progress = min(
            max(progress, 0.0),
            1.0,
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    loss_kwargs = {
        "pred_center_size_weight": (
            args.pred_center_size_weight
        ),
        "size_consistency_weight": (
            args.size_consistency_weight
        ),
        "decoded_box_weight": (
            args.decoded_box_weight
        ),
        "softargmax_beta": (
            args.softargmax_beta
        ),
        "detach_pred_center": (
            args.detach_pred_center
        ),
    }
    config = vars(args) | {
        "experiment": "V1a.2-B",
        "change_from_v1a2a": (
            "normalized decoded-corner SmoothL1 "
            "on predicted soft center and predicted-center size"
        ),
        "loss_kwargs": loss_kwargs,
        "device_resolved": str(device),
        "trainable_parameters": count_parameters(model),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
    }
    save_json(
        output_dir / "config.json",
        config,
    )
    print(json.dumps(config, indent=2))

    history: list[dict] = []
    best_prompt = -float("inf")
    best_continuous = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        model.train()
        sums: defaultdict[str, float] = (
            defaultdict(float)
        )
        batches = 0
        progress = tqdm(
            train_loader,
            desc=(
                f"V1a.2-B epoch "
                f"{epoch:02d}/{args.epochs:02d}"
            ),
        )

        for batch_cpu in progress:
            batch = move_batch_to_device(
                batch_cpu,
                device,
            )
            optimizer.zero_grad(
                set_to_none=True
            )

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
                    f"Non-finite loss at epoch "
                    f"{epoch}: {metrics}"
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
                box=f"{metrics['decoded_box_loss']:.4f}",
                box_iou=(
                    f"{metrics['decoded_box_iou_mean']:.3f}"
                ),
            )

        train_metrics = {
            f"train_{key}": value / max(batches, 1)
            for key, value in sums.items()
        }
        val_metrics, _ = evaluate_model(
            ema.module,
            val_loader,
            device,
            expansion_scale=(
                args.metric_expansion_scale
            ),
            presence_threshold=(
                args.presence_threshold
            ),
            loss_kwargs=loss_kwargs,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": (
                epoch_learning_rate
            ),
            **train_metrics,
            **val_metrics,
        }
        history.append(row)
        save_json(
            output_dir / "history.json",
            history,
        )

        checkpoint = {
            "epoch": epoch,
            "experiment": "V1a.2-B",
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "scheduler_state": (
                scheduler.state_dict()
            ),
            "config": config,
            "metrics": row,
        }
        epoch_path = (
            checkpoint_dir
            / f"epoch_{epoch:03d}.pt"
        )
        torch.save(checkpoint, epoch_path)
        torch.save(
            checkpoint,
            output_dir / "last.pt",
        )

        prompt_score = float(
            row["val_prompt_success_rate"]
        )
        continuous_score = float(
            row["val_continuous_score"]
        )
        improved = False

        if prompt_score > best_prompt:
            best_prompt = prompt_score
            torch.save(
                checkpoint,
                output_dir / "best_prompt.pt",
            )
            improved = True

        if continuous_score > best_continuous:
            best_continuous = (
                continuous_score
            )
            torch.save(
                checkpoint,
                output_dir / "best_continuous.pt",
            )
            improved = True

        epochs_without_improvement = (
            0
            if improved
            else epochs_without_improvement + 1
        )

        print(json.dumps(row, indent=2))
        print(
            f"Best prompt={best_prompt:.4f} | "
            f"best continuous={best_continuous:.4f} | "
            f"patience={epochs_without_improvement}/"
            f"{args.early_stopping_patience}"
        )

        if backup_dir:
            copy_training_state(
                output_dir,
                backup_dir,
            )
            backup_checkpoint_dir = (
                backup_dir / "checkpoints"
            )
            backup_checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                epoch_path,
                backup_checkpoint_dir
                / epoch_path.name,
            )

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print("Early stopping triggered.")
            break

    print(
        "V1a.2-B training completed: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
