from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a2b.losses import (
    compute_dual_loss,
    decode_normalized_boxes,
    heatmap_focal_loss,
)


def synthetic_batch(
    batch_size: int,
    height: int,
    width: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    outputs = {
        "presence_logits": torch.randn(
            batch_size,
            2,
            requires_grad=True,
        ),
        "heatmap_logits": torch.randn(
            batch_size,
            2,
            height,
            width,
            requires_grad=True,
        ),
        "log_size_map": torch.randn(
            batch_size,
            4,
            height,
            width,
            requires_grad=True,
        ),
        "offset_logits": torch.randn(
            batch_size,
            4,
            height,
            width,
            requires_grad=True,
        ),
    }

    heatmap = torch.zeros(
        batch_size,
        2,
        height,
        width,
    )
    heatmap[:, :, 3, 4] = 1.0

    size = torch.tensor(
        [[[0.20, 0.25], [0.18, 0.22]]]
        * batch_size
    )
    batch = {
        "presence": torch.ones(
            batch_size,
            2,
        ),
        "heatmap": heatmap,
        "center": torch.tensor(
            [[[0.56, 0.44], [0.32, 0.67]]]
            * batch_size
        ),
        "center_index": torch.tensor(
            [[[4, 3], [2, 5]]]
            * batch_size
        ),
        "offset": torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5]]]
            * batch_size
        ),
        "size": size,
        "log_size": torch.log(size),
        "relative_area": torch.tensor(
            [[0.20, 0.60]]
            * batch_size
        ),
    }
    return outputs, batch


def test_box_decode() -> None:
    center = torch.tensor(
        [[[0.5, 0.5]]]
    )
    size = torch.tensor(
        [[[0.2, 0.4]]]
    )
    box = decode_normalized_boxes(
        center,
        size,
    )
    expected = torch.tensor(
        [[[0.4, 0.3, 0.6, 0.7]]]
    )
    assert torch.allclose(
        box,
        expected,
        atol=1e-6,
    ), box


def test_saturated_amp_focal() -> None:
    if not torch.cuda.is_available():
        return
    logits = torch.tensor(
        [[[[100.0, -100.0],
           [100.0, -100.0]]]],
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[[1.0, 0.0],
           [0.0, 0.0]]]],
        device="cuda",
        dtype=torch.float16,
    )
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        loss = heatmap_focal_loss(
            logits,
            target,
        )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_decoded_box_backward() -> None:
    torch.manual_seed(7)
    outputs, batch = synthetic_batch(
        batch_size=2,
        height=8,
        width=8,
    )
    loss, metrics = compute_dual_loss(
        outputs,
        batch,
        pred_center_size_weight=0.5,
        size_consistency_weight=0.0,
        decoded_box_weight=0.5,
        detach_pred_center=True,
    )
    assert torch.isfinite(loss), metrics
    assert metrics["decoded_box_loss"] >= 0
    assert 0 <= metrics["decoded_box_iou_mean"] <= 1
    loss.backward()

    assert (
        outputs["heatmap_logits"].grad
        is not None
    )
    assert (
        outputs["heatmap_logits"]
        .grad.abs()
        .sum()
        > 0
    )
    assert (
        outputs["log_size_map"].grad
        is not None
    )
    assert (
        outputs["log_size_map"]
        .grad.abs()
        .sum()
        > 0
    )


def main() -> None:
    test_box_decode()
    test_saturated_amp_focal()
    test_decoded_box_backward()
    print("V1a.2-B LOSS TESTS PASSED")


if __name__ == "__main__":
    main()
