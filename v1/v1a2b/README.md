# V1a.2-B — Decoded-box supervision

V1a.2-B extends V1a.2-A with one controlled change: a joint loss on the decoded
bounding-box corners. The architecture, data split, sampler, augmentation,
optimizer, EMA policy, and frozen MedSAM evaluation protocol remain unchanged.

## Motivation

V1a.2-A reduced the isolated size contribution but did not reduce center-size
interaction. Supervising center and size independently does not directly
optimize the final box used by MedSAM.

V1a.2-B therefore decodes the predicted soft center and predicted-center sampled
size into normalized corners and compares them with the ground-truth box.

## Method

The V1a.2-A losses are preserved:

```text
pred_center_size_weight = 0.5
size_consistency_weight = 0.0
softargmax_beta         = 12.0
detach_pred_center      = true
```

The new term is:

```text
L_box = SmoothL1(
    decode(predicted_soft_center, predicted_center_sampled_size),
    decode(ground_truth_center, ground_truth_size)
)

decoded_box_weight = 0.5
```

The predicted center is detached only for the size-map sampling location. The
decoded-box loss can still update the center through the decoded coordinates and
update the size map through the sampled values.

No GIoU loss, small-area weighting, presence-loss change, or sampler change was
included in this ablation.

## Gates

Before full training, V1a.2-B passed:

- unit tests for decoding, finite FP16 focal loss, and gradient flow;
- deterministic small-sample overfit;
- five-epoch smoke training.

## Training configuration

```text
input mode          = 2.5D
input size          = 256
heatmap size        = 64
base channels       = 24
optimizer           = AdamW
learning rate       = 3e-4
weight decay        = 1e-4
EMA decay           = 0.995
maximum epochs      = 40
patience            = 10
seed                = 42
best continuous     = epoch 4 EMA
checkpoint Pareto   = epochs 4, 5, 11, 14
```

Training stopped at epoch 14. The positive-slice checkpoint selector chose epoch
4, but the later negative-aware checkpoint analysis selected epoch 14 for
automatic use.

## Initial positive-slice result

Epoch 4 locked configuration:

```text
variant           = smooth
max_gap           = 0
left threshold    = 0.10
right threshold   = 0.20
expansion         = 1.25
```

| Metric | Result |
|---|---:|
| Positive-slice Dice | 0.8262 |
| Positive-case Dice | 0.8419 |
| Oracle gap | 0.0450 |
| 00-10% area Dice | 0.3150 |
| Presence recall | 1.0000 |
| Side-level presence FPR | 0.8060 |

This was the strongest positive-slice segmentation result, but the operating
point produced excessive false positives on fully negative slices.

## Fixed-scale geometry result

All models were reevaluated with ground-truth presence and expansion fixed at
`1.20`.

| Metric | V1a.1 | V1a.2-A | V1a.2-B |
|---|---:|---:|---:|
| M0: auto center + auto size | 0.8361 | 0.8406 | **0.8416** |
| Center gain | 0.0149 | 0.0130 | 0.0138 |
| Size gain | 0.0096 | 0.0044 | 0.0085 |
| Joint box gap | 0.0347 | 0.0302 | **0.0292** |
| Center-size interaction | 0.0102 | 0.0128 | **0.0069** |

The decoded-box hypothesis is supported: V1a.2-B has the best M0, the smallest
joint box gap, and the lowest center-size interaction.

## Negative-aware checkpoint Pareto

Each checkpoint received an independent validation-only calibration and frozen
MedSAM evaluation.

| Epoch | All-slice Dice | Case-volume Dice | Negative specificity | Positive-slice Dice |
|---:|---:|---:|---:|---:|
| 4 | 0.3060 | 0.7534 | 0.0335 | **0.8262** |
| 5 | 0.3996 | 0.7677 | 0.1791 | 0.8206 |
| 11 | 0.7415 | 0.8485 | 0.7121 | 0.7981 |
| 14 | **0.7725** | **0.8508** | **0.7598** | 0.7972 |

Epoch 14 is the preferred V1a.2-B checkpoint for fully automatic evaluation:

```text
variant           = smooth
max_gap           = 2
left threshold    = 0.01
right threshold   = 0.01
expansion         = 1.15
presence recall   = 0.9623
side-level FPR    = 0.1991
```

It gives up some positive-slice Dice but dramatically improves negative-slice
behavior.

## Current conclusion

V1a.2-B should be retained as the leading geometry design. It is not yet the
best complete automatic operating point by case-volume Dice: V1a.2-A locked
scores `0.8556`, versus `0.8508` for V1a.2-B epoch 14.

The next development stage should focus on presence detection, negative-slice
suppression, and kidney onset/offset behavior rather than adding more box-loss
terms.

## Directory layout

```text
v1/v1a2b/
├── DESIGN.md
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_v1a2b.py
│   ├── predict_v1a2b.py
│   ├── overfit_v1a2b.py
│   └── test_v1a2b_loss.py
├── v1a2b/
│   ├── losses.py
│   ├── runtime.py
│   ├── model.py
│   └── supporting modules
├── docs/
│   ├── V1A2_VALIDATION_STATUS.md
│   └── RESUME_FOLLOWUP_VALIDATION.md
└── followup_validation/
    ├── tools/
    └── results/
```

See [follow-up validation](followup_validation/README.md) for the completed
negative-aware, fixed-scale, and checkpoint-Pareto protocol.
