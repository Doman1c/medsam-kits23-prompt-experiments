from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.data import DualKidneyDataset
from v1a1.inference import decode_dual_outputs, smooth_case_side
from v1a1.model import DualKidneyPromptNet
from v1a1.runtime import decoded_batch_to_rows, move_batch_to_device
from v1a1.utils import resolve_device, set_seed


def add_smooth_predictions(frame: pd.DataFrame, threshold: float, max_gap: int) -> pd.DataFrame:
    frame = frame.sort_values(["case_id", "slice_idx"]).copy()
    for side in ("left", "right"):
        for column in (
            f"{side}_smooth_presence_probability",
            f"{side}_smooth_presence",
            f"{side}_smooth_center_x_norm",
            f"{side}_smooth_center_y_norm",
            f"{side}_smooth_width_norm",
            f"{side}_smooth_height_norm",
            f"{side}_smooth_x1",
            f"{side}_smooth_y1",
            f"{side}_smooth_x2",
            f"{side}_smooth_y2",
        ):
            frame[column] = 0.0

    for _, index_values in frame.groupby("case_id", sort=False).groups.items():
        indices = list(index_values)
        case = frame.loc[indices].sort_values("slice_idx")
        indices = case.index.tolist()
        for side in ("left", "right"):
            probability = case[f"{side}_raw_presence_probability"].to_numpy(dtype=float)
            centers = case[[f"{side}_raw_center_x_norm", f"{side}_raw_center_y_norm"]].to_numpy(dtype=float)
            sizes = case[[f"{side}_raw_width_norm", f"{side}_raw_height_norm"]].to_numpy(dtype=float)
            probability_smooth, present, center_smooth, size_smooth = smooth_case_side(
                probability,
                centers,
                sizes,
                threshold=threshold,
                max_gap=max_gap,
            )
            for local_i, frame_index in enumerate(indices):
                image_width = float(frame.at[frame_index, "image_width"])
                image_height = float(frame.at[frame_index, "image_height"])
                cx = center_smooth[local_i, 0] * image_width
                cy = center_smooth[local_i, 1] * image_height
                width = size_smooth[local_i, 0] * image_width
                height = size_smooth[local_i, 1] * image_height
                box = (
                    max(0.0, cx - width / 2.0),
                    max(0.0, cy - height / 2.0),
                    min(image_width, cx + width / 2.0),
                    min(image_height, cy + height / 2.0),
                )
                frame.at[frame_index, f"{side}_smooth_presence_probability"] = probability_smooth[local_i]
                frame.at[frame_index, f"{side}_smooth_presence"] = int(present[local_i])
                frame.at[frame_index, f"{side}_smooth_center_x_norm"] = center_smooth[local_i, 0]
                frame.at[frame_index, f"{side}_smooth_center_y_norm"] = center_smooth[local_i, 1]
                frame.at[frame_index, f"{side}_smooth_width_norm"] = size_smooth[local_i, 0]
                frame.at[frame_index, f"{side}_smooth_height_norm"] = size_smooth[local_i, 1]
                for key, value in zip(("x1", "y1", "x2", "y2"), box):
                    frame.at[frame_index, f"{side}_smooth_{key}"] = value
    for side in ("left", "right"):
        frame[f"{side}_smooth_presence"] = frame[f"{side}_smooth_presence"].astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw and z-smoothed dual-kidney boxes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    set_seed(int(config.get("seed", 42)), deterministic=True)
    input_mode = config.get("input_mode", "2.5d")
    dataset = DualKidneyDataset(
        args.manifest,
        args.split,
        input_size=int(config.get("input_size", 256)),
        heatmap_size=int(config.get("heatmap_size", 64)),
        input_mode=input_mode,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = DualKidneyPromptNet(
        in_channels=1 if input_mode == "2d" else 3,
        base_channels=int(config.get("base_channels", 24)),
    ).to(device)
    state_key = "ema_state" if args.weights == "ema" and "ema_state" in checkpoint else "model_state"
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    rows = []
    with torch.no_grad():
        for batch_cpu in tqdm(loader, desc=f"Predicting {args.split}"):
            batch = move_batch_to_device(batch_cpu, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                outputs = model(batch["image"])
            decoded = decode_dual_outputs(
                outputs,
                batch["original_width"],
                batch["original_height"],
                presence_threshold=args.presence_threshold,
            )
            rows.extend(decoded_batch_to_rows(batch_cpu, decoded))
    frame = add_smooth_predictions(pd.DataFrame(rows), args.presence_threshold, args.max_gap)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} predictions: {output}")


if __name__ == "__main__":
    main()
