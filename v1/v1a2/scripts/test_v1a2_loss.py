from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a2.losses import compute_dual_loss, sample_map_at_normalized_centers


def main() -> None:
    feature = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    center = torch.tensor([[[0.75, 0.25]]])
    sampled = sample_map_at_normalized_centers(feature, center)
    assert torch.allclose(sampled, torch.tensor([[[2.0]]]), atol=1e-6), sampled

    torch.manual_seed(7)
    batch_size, height, width = 2, 8, 8
    outputs = {
        "presence_logits": torch.randn(batch_size, 2, requires_grad=True),
        "heatmap_logits": torch.randn(batch_size, 2, height, width, requires_grad=True),
        "log_size_map": torch.randn(batch_size, 4, height, width, requires_grad=True),
        "offset_logits": torch.randn(batch_size, 4, height, width, requires_grad=True),
    }
    heatmap = torch.zeros(batch_size, 2, height, width)
    heatmap[:, :, 3, 4] = 1.0
    sizes = torch.tensor([[[0.20, 0.25], [0.18, 0.22]]] * batch_size)
    batch = {
        "presence": torch.ones(batch_size, 2),
        "heatmap": heatmap,
        "center": torch.tensor([[[0.56, 0.44], [0.32, 0.67]]] * batch_size),
        "center_index": torch.tensor([[[4, 3], [2, 5]]] * batch_size),
        "offset": torch.tensor([[[0.5, 0.5], [0.5, 0.5]]] * batch_size),
        "size": sizes,
        "log_size": torch.log(sizes),
        "relative_area": torch.tensor([[0.20, 0.60]] * batch_size),
    }
    loss, metrics = compute_dual_loss(outputs, batch, pred_center_size_weight=0.5)
    assert torch.isfinite(loss), metrics
    loss.backward()
    assert outputs["log_size_map"].grad is not None
    assert outputs["log_size_map"].grad.abs().sum() > 0
    assert outputs["heatmap_logits"].grad is not None
    assert outputs["heatmap_logits"].grad.abs().sum() > 0
    for key in ("size_gt_center_loss", "size_pred_center_loss", "size_consistency_loss"):
        assert key in metrics
    print("V1a.2-A LOSS TESTS PASSED")


if __name__ == "__main__":
    main()
