from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable CenterNet-style focal loss under AMP."""
    logits_float = logits.float()
    target_float = target.float()

    prediction = torch.sigmoid(logits_float).clamp(
        min=1e-5,
        max=1.0 - 1e-5,
    )
    positive = target_float.eq(1.0).float()
    negative = target_float.lt(1.0).float()
    negative_weight = (1.0 - target_float).pow(4)

    positive_loss = (
        -torch.log(prediction)
        * (1.0 - prediction).pow(2)
        * positive
    )
    negative_loss = (
        -torch.log1p(-prediction)
        * prediction.pow(2)
        * negative_weight
        * negative
    )

    positive_count = positive.sum()
    normalized = (
        positive_loss.sum() + negative_loss.sum()
    ) / positive_count.clamp_min(1.0)
    negative_only = negative_loss.mean()

    return torch.where(
        positive_count > 0,
        normalized,
        negative_only,
    )


def gather_center_values(
    feature_map: torch.Tensor,
    center_index: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, side, value, H, W] at integer GT centers."""
    batch, sides, values, height, width = feature_map.shape
    x = center_index[..., 0].clamp(0, width - 1)
    y = center_index[..., 1].clamp(0, height - 1)
    flat_index = (
        (y * width + x)
        .view(batch, sides, 1, 1)
        .expand(-1, -1, values, 1)
    )
    flattened = feature_map.view(
        batch,
        sides,
        values,
        height * width,
    )
    return flattened.gather(-1, flat_index).squeeze(-1)


def softargmax_centers(
    logits: torch.Tensor,
    beta: float = 12.0,
) -> torch.Tensor:
    """Differentiable normalized centers [B, side, xy]."""
    batch, sides, height, width = logits.shape
    probability = torch.softmax(
        logits.view(batch, sides, -1) * float(beta),
        dim=-1,
    )
    yy, xx = torch.meshgrid(
        torch.arange(
            height,
            device=logits.device,
            dtype=logits.dtype,
        ),
        torch.arange(
            width,
            device=logits.device,
            dtype=logits.dtype,
        ),
        indexing="ij",
    )
    x_grid = (
        (xx.reshape(-1) + 0.5) / width
    ).view(1, 1, -1)
    y_grid = (
        (yy.reshape(-1) + 0.5) / height
    ).view(1, 1, -1)
    pred_x = (probability * x_grid).sum(dim=-1)
    pred_y = (probability * y_grid).sum(dim=-1)
    return torch.stack([pred_x, pred_y], dim=-1)


def sample_map_at_normalized_centers(
    feature_map: torch.Tensor,
    centers: torch.Tensor,
    *,
    detach_centers: bool = True,
) -> torch.Tensor:
    """
    Bilinearly sample [B,S,V,H,W] at normalized [B,S,2] centers.

    align_corners=False is compatible with centers represented as
    (index + 0.5) / size.
    """
    if feature_map.ndim != 5:
        raise ValueError(
            f"feature_map must be [B,S,V,H,W], got {feature_map.shape}"
        )
    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError(
            f"centers must be [B,S,2], got {centers.shape}"
        )

    batch, sides, values, height, width = feature_map.shape
    if centers.shape[:2] != (batch, sides):
        raise ValueError(
            "feature_map and centers dimensions do not match"
        )

    sample_centers = (
        centers.detach()
        if detach_centers
        else centers
    ).clamp(0.0, 1.0)

    maps = feature_map.reshape(
        batch * sides,
        values,
        height,
        width,
    )
    grid = (
        sample_centers.reshape(
            batch * sides,
            1,
            1,
            2,
        )
        * 2.0
        - 1.0
    )
    sampled = F.grid_sample(
        maps,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[:, :, 0, 0].reshape(
        batch,
        sides,
        values,
    )


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (
        values * mask
    ).sum() / mask.sum().clamp_min(1.0)


def area_weighted_present_mean(
    values: torch.Tensor,
    area_weight: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    weight = (
        area_weight.to(values.dtype)
        * present.to(values.dtype)
    )
    return (
        values * weight
    ).sum() / weight.sum().clamp_min(1.0)


def decode_normalized_boxes(
    centers: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Decode normalized center/size to [x1,y1,x2,y2].

    centers: [B,S,2], sizes: [B,S,2].
    """
    half = sizes.clamp(1e-4, 1.5) / 2.0
    top_left = centers - half
    bottom_right = centers + half
    return torch.cat(
        [top_left, bottom_right],
        dim=-1,
    ).clamp(0.0, 1.0)


def aligned_box_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Aligned IoU for [...,4] boxes, used only as a metric."""
    tl = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    br = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    wh = (br - tl).clamp_min(0.0)
    intersection = wh[..., 0] * wh[..., 1]

    area_a_wh = (
        boxes_a[..., 2:] - boxes_a[..., :2]
    ).clamp_min(0.0)
    area_b_wh = (
        boxes_b[..., 2:] - boxes_b[..., :2]
    ).clamp_min(0.0)
    area_a = area_a_wh[..., 0] * area_a_wh[..., 1]
    area_b = area_b_wh[..., 0] * area_b_wh[..., 1]
    union = area_a + area_b - intersection
    return intersection / union.clamp_min(1e-8)


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
    pred_center_size_weight: float = 0.5,
    size_consistency_weight: float = 0.0,
    decoded_box_weight: float = 0.5,
    softargmax_beta: float = 12.0,
    detach_pred_center: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    V1a.2-B loss.

    Relative to V1a.2-A, the only new optimized term is normalized
    decoded-corner SmoothL1:

        L_box = SmoothL1(
            decode(predicted soft center, predicted-center size),
            decode(GT center, GT size)
        )

    The decoded-box loss supervises center and size jointly. Size-map
    sampling still detaches the center by default, so the new box loss
    changes the center through the decoded center coordinates, not through
    grid-sampling-location gradients.
    """
    alpha = float(pred_center_size_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "pred_center_size_weight must be in [0,1]"
        )
    if decoded_box_weight < 0:
        raise ValueError(
            "decoded_box_weight must be non-negative"
        )

    presence_target = batch["presence"]
    present = presence_target > 0.5

    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"],
        presence_target,
    )
    heatmap_loss = heatmap_focal_loss(
        outputs["heatmap_logits"],
        batch["heatmap"],
    )

    batch_size, _, height, width = (
        outputs["heatmap_logits"].shape
    )
    log_size_map = outputs["log_size_map"].view(
        batch_size,
        2,
        2,
        height,
        width,
    )
    offset_map = torch.sigmoid(
        outputs["offset_logits"].view(
            batch_size,
            2,
            2,
            height,
            width,
        )
    )

    gt_center_log_size = gather_center_values(
        log_size_map,
        batch["center_index"],
    )
    predicted_offset = gather_center_values(
        offset_map,
        batch["center_index"],
    )

    predicted_center = softargmax_centers(
        outputs["heatmap_logits"],
        beta=softargmax_beta,
    )
    pred_center_log_size = (
        sample_map_at_normalized_centers(
            log_size_map,
            predicted_center,
            detach_centers=detach_pred_center,
        )
    )

    gt_size_element = F.smooth_l1_loss(
        gt_center_log_size,
        batch["log_size"],
        reduction="none",
    ).mean(dim=-1)
    pred_size_element = F.smooth_l1_loss(
        pred_center_log_size,
        batch["log_size"],
        reduction="none",
    ).mean(dim=-1)

    area_weight = (
        batch["relative_area"]
        .clamp(0.10, 1.0)
        .sqrt()
    )
    size_gt_center_loss = area_weighted_present_mean(
        gt_size_element,
        area_weight,
        present,
    )
    size_pred_center_loss = area_weighted_present_mean(
        pred_size_element,
        area_weight,
        present,
    )
    size_loss = (
        (1.0 - alpha) * size_gt_center_loss
        + alpha * size_pred_center_loss
    )

    consistency_element = F.smooth_l1_loss(
        pred_center_log_size,
        gt_center_log_size.detach(),
        reduction="none",
    ).mean(dim=-1)
    size_consistency_loss = area_weighted_present_mean(
        consistency_element,
        area_weight,
        present,
    )

    offset_element = F.smooth_l1_loss(
        predicted_offset,
        batch["offset"],
        reduction="none",
    ).mean(dim=-1)
    offset_loss = masked_mean(
        offset_element,
        present,
    )

    predicted_size = (
        pred_center_log_size
        .exp()
        .clamp(1e-4, 1.5)
    )
    under_size = F.relu(
        batch["size"] - predicted_size
    ).mean(dim=-1)
    under_size_loss = masked_mean(
        under_size,
        present,
    )

    coordinate_element = F.smooth_l1_loss(
        predicted_center,
        batch["center"],
        reduction="none",
    ).mean(dim=-1)
    coordinate_loss = masked_mean(
        coordinate_element,
        present,
    )

    predicted_boxes = decode_normalized_boxes(
        predicted_center,
        predicted_size,
    )
    target_boxes = decode_normalized_boxes(
        batch["center"],
        batch["size"],
    )
    decoded_box_element = F.smooth_l1_loss(
        predicted_boxes,
        target_boxes,
        reduction="none",
    ).mean(dim=-1)
    decoded_box_loss = area_weighted_present_mean(
        decoded_box_element,
        area_weight,
        present,
    )

    decoded_iou = aligned_box_iou(
        predicted_boxes.detach(),
        target_boxes.detach(),
    )
    decoded_box_iou_mean = masked_mean(
        decoded_iou,
        present,
    )

    both_present = present[:, 0] & present[:, 1]
    if both_present.any():
        order_loss = F.relu(
            predicted_center[both_present, 0, 0]
            - predicted_center[both_present, 1, 0]
            + 0.05
        ).mean()
    else:
        order_loss = (
            outputs["heatmap_logits"].sum() * 0.0
        )

    total = (
        presence_weight * presence_loss
        + heatmap_weight * heatmap_loss
        + coordinate_weight * coordinate_loss
        + size_weight * size_loss
        + offset_weight * offset_loss
        + under_size_weight * under_size_loss
        + order_weight * order_loss
        + size_consistency_weight
        * size_consistency_loss
        + decoded_box_weight
        * decoded_box_loss
    )

    metrics = {
        "loss": float(total.detach()),
        "presence_loss": float(
            presence_loss.detach()
        ),
        "heatmap_loss": float(
            heatmap_loss.detach()
        ),
        "coordinate_loss": float(
            coordinate_loss.detach()
        ),
        "size_loss": float(size_loss.detach()),
        "size_gt_center_loss": float(
            size_gt_center_loss.detach()
        ),
        "size_pred_center_loss": float(
            size_pred_center_loss.detach()
        ),
        "size_consistency_loss": float(
            size_consistency_loss.detach()
        ),
        "offset_loss": float(
            offset_loss.detach()
        ),
        "under_size_loss": float(
            under_size_loss.detach()
        ),
        "decoded_box_loss": float(
            decoded_box_loss.detach()
        ),
        "decoded_box_iou_mean": float(
            decoded_box_iou_mean.detach()
        ),
        "order_loss": float(
            order_loss.detach()
        ),
        "pred_center_size_weight": alpha,
        "decoded_box_weight": float(
            decoded_box_weight
        ),
    }
    return total, metrics
