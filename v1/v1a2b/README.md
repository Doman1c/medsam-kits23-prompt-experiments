# V1a.2-A Starter

V1a.2-A keeps the V1a.1 architecture and changes only size supervision:

```text
L_size = (1-alpha) * L_size_at_GT_center
       + alpha       * L_size_at_predicted_center
```

Locked first-ablation defaults:

```text
alpha                   = 0.5
detach_pred_center      = true
size_consistency_weight = 0.0
softargmax_beta         = 12.0
```

The predicted center is detached so the new size term trains the size map without sending size-loss gradients into the heatmap head.
