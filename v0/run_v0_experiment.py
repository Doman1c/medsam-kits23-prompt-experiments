from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry


DEFAULT_MODES = ["box", "box_pos", "box_pos_neg", "box_mask", "all"]
VALID_MODES = set(DEFAULT_MODES)


@dataclass
class PromptSet:
    box_xyxy: np.ndarray
    positive_point_xy: np.ndarray
    negative_point_xy: np.ndarray
    coarse_mask_logits: np.ndarray


@dataclass
class SelectedPrompts:
    box_xyxy: np.ndarray | None = None
    point_coords_xy: np.ndarray | None = None
    point_labels: np.ndarray | None = None
    mask_logits: np.ndarray | None = None


@dataclass
class MedSAMOutput:
    mask: np.ndarray
    predicted_iou: float


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(2.0 * intersection / denominator)


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(intersection / union)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    labels, count = ndi.label(
        binary,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return binary
    areas = ndi.sum(binary, labels, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)


def prepare_target(
    label_map: np.ndarray,
    target_id: int,
    component_mode: str,
) -> np.ndarray:
    target = (label_map == int(target_id)).astype(np.uint8)
    if component_mode == "largest":
        return largest_connected_component(target)
    if component_mode == "full":
        return target
    raise ValueError(f"Unknown component mode: {component_mode}")


def mask_to_box(mask: np.ndarray, padding: int = 2) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot create prompts from an empty mask.")
    height, width = mask.shape
    return np.asarray(
        [
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(width - 1, int(xs.max()) + padding),
            min(height - 1, int(ys.max()) + padding),
        ],
        dtype=np.float32,
    )


def positive_point(mask: np.ndarray) -> np.ndarray:
    distance = ndi.distance_transform_edt(mask > 0)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return np.asarray([x, y], dtype=np.float32)


def negative_point(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Choose a background point near the target boundary."""
    binary = mask > 0
    height, width = binary.shape
    iterations = max(8, int(round(max(height, width) * 0.03)))
    dilated = ndi.binary_dilation(binary, iterations=iterations)
    ring = np.logical_and(dilated, np.logical_not(binary))

    x1, y1, x2, y2 = box
    margin = max(10, iterations * 2)
    roi = np.zeros_like(binary, dtype=bool)
    roi[
        max(0, int(y1) - margin) : min(height, int(y2) + margin + 1),
        max(0, int(x1) - margin) : min(width, int(x2) + margin + 1),
    ] = True
    candidates = np.logical_and(ring, roi)
    if not candidates.any():
        candidates = np.logical_not(binary)

    distance_to_target = ndi.distance_transform_edt(np.logical_not(binary))
    scores = np.where(candidates, distance_to_target, np.inf)
    y, x = np.unravel_index(int(np.argmin(scores)), scores.shape)
    return np.asarray([x, y], dtype=np.float32)


def coarse_mask_logits(mask: np.ndarray) -> np.ndarray:
    """Downsample to 64 and expand to the 256x256 SAM mask-input size."""
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    coarse = F.interpolate(
        tensor, size=(64, 64), mode="bilinear", align_corners=False
    )
    coarse = F.interpolate(
        coarse, size=(256, 256), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    probability = np.clip(coarse, 1e-4, 1.0 - 1e-4)
    return np.log(probability / (1.0 - probability)).astype(np.float32)


def generate_prompt_set(mask: np.ndarray) -> PromptSet:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    box = mask_to_box(mask)
    return PromptSet(
        box_xyxy=box,
        positive_point_xy=positive_point(mask),
        negative_point_xy=negative_point(mask, box),
        coarse_mask_logits=coarse_mask_logits(mask),
    )


def select_prompt_mode(prompts: PromptSet, mode: str) -> SelectedPrompts:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode}; choose from {sorted(VALID_MODES)}")
    if mode == "box":
        return SelectedPrompts(box_xyxy=prompts.box_xyxy)
    if mode == "box_pos":
        return SelectedPrompts(
            box_xyxy=prompts.box_xyxy,
            point_coords_xy=prompts.positive_point_xy[None],
            point_labels=np.asarray([1], dtype=np.int64),
        )
    if mode == "box_pos_neg":
        return SelectedPrompts(
            box_xyxy=prompts.box_xyxy,
            point_coords_xy=np.stack(
                [prompts.positive_point_xy, prompts.negative_point_xy]
            ),
            point_labels=np.asarray([1, 0], dtype=np.int64),
        )
    if mode == "box_mask":
        return SelectedPrompts(
            box_xyxy=prompts.box_xyxy,
            mask_logits=prompts.coarse_mask_logits,
        )
    return SelectedPrompts(
        box_xyxy=prompts.box_xyxy,
        point_coords_xy=np.stack(
            [prompts.positive_point_xy, prompts.negative_point_xy]
        ),
        point_labels=np.asarray([1, 0], dtype=np.int64),
        mask_logits=prompts.coarse_mask_logits,
    )


class MultiPromptMedSAM:
    image_size = 1024

    def __init__(self, checkpoint: str, device: str | None = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("MedSAM device:", self.device)

    @torch.inference_mode()
    def encode_image(
        self, image: np.ndarray
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got {image.shape}")
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        minimum, maximum = float(image.min()), float(image.max())
        image = (
            (image - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(image)
        )
        original_shape = (int(image.shape[0]), int(image.shape[1]))
        tensor = torch.from_numpy(image)[None, None].to(self.device)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        ).repeat(1, 3, 1, 1)
        return self.model.image_encoder(tensor), original_shape

    def scale_box(
        self, box: np.ndarray, original_shape: tuple[int, int]
    ) -> torch.Tensor:
        height, width = original_shape
        scaled = np.asarray(box, dtype=np.float32).copy()
        scaled[[0, 2]] *= self.image_size / width
        scaled[[1, 3]] *= self.image_size / height
        return torch.from_numpy(scaled)[None].to(self.device)

    def scale_points(
        self, points: np.ndarray, original_shape: tuple[int, int]
    ) -> torch.Tensor:
        height, width = original_shape
        scaled = np.asarray(points, dtype=np.float32).copy()
        scaled[:, 0] *= self.image_size / width
        scaled[:, 1] *= self.image_size / height
        return torch.from_numpy(scaled)[None].to(self.device)

    @torch.inference_mode()
    def decode(
        self,
        image_embedding: torch.Tensor,
        original_shape: tuple[int, int],
        prompts: SelectedPrompts,
    ) -> MedSAMOutput:
        box_tensor = (
            self.scale_box(prompts.box_xyxy, original_shape)
            if prompts.box_xyxy is not None
            else None
        )

        point_input = None
        if prompts.point_coords_xy is not None:
            coords = self.scale_points(prompts.point_coords_xy, original_shape)
            labels = torch.from_numpy(
                np.asarray(prompts.point_labels, dtype=np.int64)
            )[None].to(self.device)
            point_input = (coords, labels)

        mask_tensor = None
        if prompts.mask_logits is not None:
            mask_tensor = torch.from_numpy(
                np.asarray(prompts.mask_logits, dtype=np.float32)
            )[None, None].to(self.device)

        sparse, dense = self.model.prompt_encoder(
            points=point_input,
            boxes=box_tensor,
            masks=mask_tensor,
        )
        low_res_masks, predicted_iou = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = self.model.postprocess_masks(
            low_res_masks,
            input_size=(self.image_size, self.image_size),
            original_size=original_shape,
        )
        mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        return MedSAMOutput(
            mask=mask,
            predicted_iou=float(predicted_iou[0, 0].cpu().item()),
        )


def draw_prompts(axis, prompts: PromptSet) -> None:
    x1, y1, x2, y2 = prompts.box_xyxy
    axis.add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
    )
    axis.scatter(*prompts.positive_point_xy, marker="*", s=100)
    axis.scatter(*prompts.negative_point_xy, marker="x", s=70)


def save_figure(
    path: Path,
    image: np.ndarray,
    target: np.ndarray,
    prompts: PromptSet,
    predictions: dict[str, np.ndarray],
) -> None:
    columns = 2 + len(predictions)
    figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(image, cmap="gray")
    draw_prompts(axes[0], prompts)
    axes[0].set_title("Image + prompts")
    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("Evaluation target")
    for axis, (mode, prediction) in zip(axes[2:], predictions.items()):
        axis.imshow(prediction, cmap="gray")
        axis.set_title(mode)
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def restrict_cases(data: pd.DataFrame, max_cases: int | None) -> pd.DataFrame:
    if max_cases is None:
        return data
    case_ids = data["case_id"].drop_duplicates().head(max_cases).tolist()
    return data[data["case_id"].isin(case_ids)].copy()


def select_samples(
    data: pd.DataFrame,
    sampling: str,
    max_samples: int | None,
    samples_per_case: int,
    seed: int,
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("No matching samples.")

    if sampling == "one_per_case":
        selected = pd.DataFrame(
            [
                frame.loc[frame["target_pixels"].idxmax()]
                for _, frame in data.groupby("case_id", sort=True)
            ]
        )
    elif sampling == "topk_per_case":
        selected = pd.concat(
            [
                frame.nlargest(min(samples_per_case, len(frame)), "target_pixels")
                for _, frame in data.groupby("case_id", sort=True)
            ],
            ignore_index=True,
        )
    elif sampling == "random":
        count = len(data) if max_samples is None else min(max_samples, len(data))
        selected = data.sample(n=count, random_state=seed)
    elif sampling == "all":
        selected = data.copy()
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")

    selected = selected.reset_index(drop=True)
    if sampling != "random" and max_samples is not None:
        selected = selected.head(max_samples).reset_index(drop=True)
    return selected


def save_results(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "per_sample_results.csv", index=False)
    summary = (
        results.groupby(["target_name", "mode"])[
            ["dice", "iou", "medsam_predicted_iou"]
        ]
        .agg(["mean", "std", "min", "median", "max", "count"])
    )
    summary.to_csv(output_dir / "summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM with multiple generated prompt combinations."
    )
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-name", choices=["kidney", "tumor", "cyst"], default=None
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument(
        "--sampling",
        choices=["one_per_case", "topk_per_case", "random", "all"],
        default="one_per_case",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--samples-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--component-mode",
        choices=["largest", "full"],
        default="largest",
    )
    parser.add_argument("--save-visualizations", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_modes = [mode for mode in args.modes if mode not in VALID_MODES]
    if invalid_modes:
        raise ValueError(f"Invalid prompt modes: {invalid_modes}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    index_path = args.processed_root / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = args.output_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)

    data = pd.read_csv(index_path)
    required = {
        "case_id",
        "slice_index",
        "npz_path",
        "target_id",
        "target_name",
        "target_pixels",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"index.csv is missing columns: {sorted(missing)}")

    if args.target_name is not None:
        data = data[data["target_name"] == args.target_name].copy()

    max_cases = None if args.max_cases <= 0 else args.max_cases
    max_samples = None if args.max_samples <= 0 else args.max_samples
    data = restrict_cases(data, max_cases)
    data = select_samples(
        data,
        sampling=args.sampling,
        max_samples=max_samples,
        samples_per_case=args.samples_per_case,
        seed=args.seed,
    )
    data.to_csv(args.output_dir / "selected_samples.csv", index=False)

    print("Sampling method:", args.sampling)
    print("Component mode:", args.component_mode)
    print("Selected cases:", data["case_id"].nunique())
    print("Selected samples:", len(data))
    print("Target:", args.target_name)
    print("Modes:", args.modes)

    medsam = MultiPromptMedSAM(str(args.checkpoint), device=args.device)
    rows: list[dict] = []
    skipped = 0

    progress = tqdm(
        enumerate(data.itertuples(index=False), start=1),
        total=len(data),
        desc="Running MedSAM",
    )
    for completed, row in progress:
        sample_path = args.processed_root / row.npz_path
        if not sample_path.exists():
            print(f"[skip] Missing file: {sample_path}")
            skipped += 1
            continue

        try:
            with np.load(sample_path) as sample:
                image = sample["image"].astype(np.float32)
                label_map = sample["label"].astype(np.uint8)

            target = prepare_target(
                label_map,
                target_id=int(row.target_id),
                component_mode=args.component_mode,
            )
            if target.sum() == 0:
                raise ValueError("empty evaluation target")

            prompts = generate_prompt_set(target)
            embedding, original_shape = medsam.encode_image(image)
            predictions: dict[str, np.ndarray] = {}

            for mode in args.modes:
                output = medsam.decode(
                    embedding,
                    original_shape,
                    select_prompt_mode(prompts, mode),
                )
                predictions[mode] = output.mask
                rows.append(
                    {
                        "case_id": row.case_id,
                        "slice_index": int(row.slice_index),
                        "target_id": int(row.target_id),
                        "target_name": row.target_name,
                        "target_pixels_index": int(row.target_pixels),
                        "target_pixels_evaluated": int(target.sum()),
                        "component_mode": args.component_mode,
                        "mode": mode,
                        "dice": dice_score(output.mask, target),
                        "iou": iou_score(output.mask, target),
                        "medsam_predicted_iou": output.predicted_iou,
                    }
                )

            if completed <= args.save_visualizations:
                save_figure(
                    visualization_dir
                    / f"{row.case_id}_z{int(row.slice_index):04d}_{row.target_name}.png",
                    image,
                    target,
                    prompts,
                    predictions,
                )

            if args.save_every > 0 and completed % args.save_every == 0:
                save_results(rows, args.output_dir)

        except Exception as error:
            print(
                f"[skip] {row.case_id}, slice {row.slice_index}: "
                f"{type(error).__name__}: {error}"
            )
            skipped += 1

    save_results(rows, args.output_dir)
    if not rows:
        raise RuntimeError("No samples were evaluated successfully.")

    results = pd.DataFrame(rows)
    print("\nMean Dice by mode:")
    print(results.groupby("mode")["dice"].mean().sort_values(ascending=False))
    unique_samples = results[
        ["case_id", "slice_index", "target_name"]
    ].drop_duplicates()
    print("\nEvaluated cases:", results["case_id"].nunique())
    print("Evaluated slice-target samples:", len(unique_samples))
    print("Skipped samples:", skipped)
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
