# V1a.2-A — Predicted-center size supervision

V1a.2-A is the first controlled ablation after the frozen V1a.1 baseline. It
keeps the network architecture, dataset split, sampler, augmentation, optimizer,
EMA policy, and inference decoder unchanged. The only intended change is how the
size map is supervised.

## Motivation

V1a.1 trains the size head at the ground-truth center but reads the size map at
the predicted heatmap peak during inference. This creates a train-inference
mismatch:

```text
training:  size = size_map[ground-truth center]
inference: size = size_map[predicted center]
```

The V1a.1 center-size decomposition showed that both isolated size error and
center-size interaction contributed to the remaining box gap.

## Method

V1a.2-A adds predicted-center sampling to the size loss:

```text
L_size = (1 - alpha) * L_size_at_GT_center
       + alpha       * L_size_at_predicted_center
```

Locked ablation settings:

```text
pred_center_size_weight = 0.5
size_consistency_weight = 0.0
softargmax_beta         = 12.0
detach_pred_center      = true
```

The predicted center is detached for the size-sampling path. The additional size
term therefore trains the size map without sending size-loss gradients back into
the heatmap head.

## Training configuration

```text
input mode        = 2.5D
input size        = 256
heatmap size      = 64
base channels     = 24
optimizer         = AdamW
learning rate     = 3e-4
weight decay      = 1e-4
EMA decay         = 0.995
maximum epochs    = 40
patience          = 10
seed              = 42
best checkpoint   = epoch 12 EMA
```

The architecture contains approximately 1.05 million parameters.

## Validation results

### Positive-slice MedSAM

Locked configuration:

```text
variant           = smooth
max_gap           = 2
left threshold    = 0.01
right threshold   = 0.01
expansion         = 1.20
```

| Metric | Result |
|---|---:|
| Slice Dice | 0.8096 |
| Case Dice | 0.8261 |
| Oracle gap | 0.0613 |
| Presence recall | 0.9702 |
| Presence FPR | 0.2252 |

A wider presence-Pareto search did not improve this operating point.

### Fixed-scale geometry decomposition

All four modes use ground-truth presence and expansion `1.20`.

| Metric | V1a.1 | V1a.2-A | Change |
|---|---:|---:|---:|
| M0: auto center + auto size | 0.8361 | 0.8406 | +0.0045 |
| Center gain | 0.0149 | 0.0130 | -0.0019 |
| Size gain | 0.0096 | 0.0044 | -0.0052 |
| Joint box gap | 0.0347 | 0.0302 | -0.0045 |
| Interaction | 0.0102 | 0.0128 | +0.0026 |

The isolated size contribution decreased substantially, supporting the original
train-inference mismatch hypothesis. Interaction did not improve, motivating the
joint decoded-box loss tested in V1a.2-B.

### Negative-aware all-slice evaluation

The follow-up evaluation covers 3,644 validation slices, including 2,390 fully
negative slices.

| Metric | Result |
|---|---:|
| All-slice Dice | 0.7666 |
| All-slice case mean Dice | 0.7756 |
| Case-volume Dice | **0.8556** |
| Negative specificity | 0.7444 |
| Negative false-mask rate | 0.2556 |

V1a.2-A is currently the strongest evaluated complete automatic operating point
by case-volume Dice.

## Interpretation

V1a.2-A succeeded at its narrow objective: predicted-center size supervision
reduced the size-related geometry gap. It did not become the best positive-slice
MedSAM model, and center-size interaction remained. The experiment should be
retained as evidence for inference-consistent size supervision, not treated as
the final architecture.

## Directory layout

```text
v1/v1a2/
├── DESIGN.md
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_v1a2.py
│   ├── predict_v1a2.py
│   ├── overfit_v1a2.py
│   └── test_v1a2_loss.py
└── v1a2/
    ├── losses.py
    ├── runtime.py
    ├── model.py
    ├── inference.py
    └── supporting modules
```

## Relationship to V1a.2-B

V1a.2-B preserves this predicted-center size supervision and adds a normalized
decoded-corner SmoothL1 loss. See [V1a.2-B](../v1a2b/README.md).
