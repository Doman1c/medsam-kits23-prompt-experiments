from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.data import load_npz


def choose_examples(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    selected = []
    groups = [
        frame[(frame.left_presence == 1) & (frame.right_presence == 1)],
        frame[(frame.left_presence == 1) & (frame.right_presence == 0)],
        frame[(frame.left_presence == 0) & (frame.right_presence == 1)],
        frame[(frame.left_area_bin == "00_10") | (frame.right_area_bin == "00_10")],
        frame[frame.has_any == 0],
    ]
    for group in groups:
        if len(group):
            selected.append(group.sample(1, random_state=42 + len(selected)))
    remaining = max(0, count - len(selected))
    if remaining:
        pool = frame.drop(index=pd.concat(selected).index if selected else [])
        if len(pool):
            selected.append(pool.sample(min(remaining, len(pool)), random_state=99))
    return pd.concat(selected).head(count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual sanity check for dual-kidney targets")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    frame = frame[frame.split == args.split]
    examples = choose_examples(frame, args.num_samples)
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (_, row) in zip(axes, examples.iterrows()):
        image, _ = load_npz(row.npz_path)
        axis.imshow(image, cmap="gray")
        for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
            if int(row[f"{side}_presence"]):
                x1, y1, x2, y2 = [row[f"{side}_{key}"] for key in ("x1", "y1", "x2", "y2")]
                axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
                axis.text(x1, y1, side, color=color, fontsize=9, backgroundcolor="black")
        axis.axvline(image.shape[1] / 2.0, linestyle="--", linewidth=0.8)
        axis.set_title(f"{row.case_id} z={int(row.slice_idx)} | L={int(row.left_presence)} R={int(row.right_presence)}")
        axis.axis("off")
    for axis in axes[len(examples):]:
        axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved sanity figure: {output}")


if __name__ == "__main__":
    main()
