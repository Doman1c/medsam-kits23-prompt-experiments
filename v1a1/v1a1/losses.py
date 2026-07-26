from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = target.eq(1.0).float()
    negative = target.lt(1.0).float()
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -torch.log(prediction) * (1.0 - prediction).pow(2) * positive
    negative_loss = (
        -torch.log(1.0 - prediction)
        * prediction.pow(2)
        * negative_weight
        * negative
    )
    count = positive.sum()
    if count > 0:
        return (positive_loss.sum() + negative_loss.sum()) / count
    return negative_loss.mean()


def gather_center_values(
    feature_map: torch.Tensor,
    center_index: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, side, value, H, W] at GT center indices."""
    batch, sides, values, height, width = feature_map.shape
    x = center_index[..., 0].clamp(0, width - 1)
    y = center_index[..., 1].clamp(0, height - 1)
    flat_index = (y * width + x).view(batch, sides, 1, 1).expand(-1, -1, values, 1)
    flattened = feature_map.view(batch, sides, values, height * width)
    return flattened.gather(-1, flat_index).squeeze(-1)


def softargmax_centers(logits: torch.Tensor, beta: float = 12.0) -> torch.Tensor:
    batch, sides, height, width = logits.shape
    probability = torch.softmax(logits.view(batch, sides, -1) * beta, dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=logits.device, dtype=logits.dtype),
        torch.arange(width, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    x_grid = ((xx.reshape(-1) + 0.5) / width).view(1, 1, -1)
    y_grid = ((yy.reshape(-1) + 0.5) / height).view(1, 1, -1)
    pred_x = (probability * x_grid).sum(dim=-1)
    pred_y = (probability * y_grid).sum(dim=-1)
    return torch.stack([pred_x, pred_y], dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_dual_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    presence_weight: float = 1.0,
    heatmap_weight: float = 2.0,
    coordinate_weight: float = 1.0,
    size_weight: float = 1.0,
    offset_weight: float = 1.0,
    under_size_weight: float = 0.5,
    order_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    presence_target = batch["presence"]
    present = presence_target > 0.5

    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], presence_target
    )
    heatmap_loss = heatmap_focal_loss(outputs["heatmap_logits"], batch["heatmap"])

    batch_size, _, height, width = outputs["heatmap_logits"].shape
    log_size_map = outputs["log_size_map"].view(batch_size, 2, 2, height, width)
    offset_map = torch.sigmoid(
        outputs["offset_logits"].view(batch_size, 2, 2, height, width)
    )
    predicted_log_size = gather_center_values(log_size_map, batch["center_index"])
    predicted_offset = gather_center_values(offset_map, batch["center_index"])

    size_element = F.smooth_l1_loss(
        predicted_log_size, batch["log_size"], reduction="none"
    ).mean(dim=-1)
    # Preserve the original v1a principle: very small edge boxes should not dominate size regression.
    area_weight = batch["relative_area"].clamp(0.10, 1.0).sqrt()
    size_loss = (
        size_element * area_weight * present.float()
    ).sum() / (area_weight * present.float()).sum().clamp_min(1.0)

    offset_element = F.smooth_l1_loss(
        predicted_offset, batch["offset"], reduction="none"
    ).mean(dim=-1)
    offset_loss = masked_mean(offset_element, present)

    predicted_size = predicted_log_size.exp().clamp(1e-4, 1.5)
    under_size = F.relu(batch["size"] - predicted_size).mean(dim=-1)
    under_size_loss = masked_mean(under_size, present)

    predicted_center = softargmax_centers(outputs["heatmap_logits"])
    coordinate_element = F.smooth_l1_loss(
        predicted_center, batch["center"], reduction="none"
    ).mean(dim=-1)
    coordinate_loss = masked_mean(coordinate_element, present)

    both_present = present[:, 0] & present[:, 1]
    if both_present.any():
        # Canonical RAS requires anatomical left center x < anatomical right center x.
        order_loss = F.relu(
            predicted_center[both_present, 0, 0]
            - predicted_center[both_present, 1, 0]
            + 0.05
        ).mean()
    else:
        order_loss = outputs["heatmap_logits"].sum() * 0.0

    total = (
        presence_weight * presence_loss
        + heatmap_weight * heatmap_loss
        + coordinate_weight * coordinate_loss
        + size_weight * size_loss
        + offset_weight * offset_loss
        + under_size_weight * under_size_loss
        + order_weight * order_loss
    )
    metrics = {
        "loss": float(total.detach()),
        "presence_loss": float(presence_loss.detach()),
        "heatmap_loss": float(heatmap_loss.detach()),
        "coordinate_loss": float(coordinate_loss.detach()),
        "size_loss": float(size_loss.detach()),
        "offset_loss": float(offset_loss.detach()),
        "under_size_loss": float(under_size_loss.detach()),
        "order_loss": float(order_loss.detach()),
    }
    return total, metrics
