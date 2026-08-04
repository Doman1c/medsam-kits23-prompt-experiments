# MedSAM Automatic Prompt Generation on KiTS23

This repository studies a lightweight, modular, and interpretable automatic
spatial-prompt generator for frozen MedSAM on the KiTS23 kidney CT dataset.
The prompt model predicts bilateral kidney presence and bounding boxes from
2.5D CT input; MedSAM itself remains frozen.

```text
2.5D CT slices
    -> bilateral presence / center / size prediction
    -> calibrated spatial prompts
    -> frozen MedSAM
    -> kidney segmentation
```

## Research objective

The project asks two related questions:

1. Can a compact prompt generator replace manually supplied MedSAM boxes?
2. Which errors dominate the automatic system: presence decisions, box center,
   box size, or center-size interaction?

Development follows controlled validation-only ablations. The frozen test split
is not used to select checkpoints, thresholds, smoothing, max-gap, expansion,
samplers, or loss variants.

## Experiment versions

| Version | Main question | Status |
|---|---|---|
| [V0](v0/) | Which MedSAM prompt combinations work best? | Complete |
| [V0.5](v0.5/) | How robust is MedSAM to prompt degradation? | Complete |
| [V1a.1](v1/v1a1/) | Can a 2.5D dual-kidney network generate automatic boxes? | Frozen test complete |
| [V1a.2-A](v1/v1a2/) | Does predicted-center size supervision repair train-inference mismatch? | Validation complete |
| [V1a.2-B](v1/v1a2b/) | Does decoded-corner supervision reduce joint center-size error? | Validation complete |
| [V1a.2 follow-up](v1/v1a2b/followup_validation/) | Which model survives fully negative slices and checkpoint-Pareto analysis? | Complete |
| [V1a.3-A](v1/v1a3/) | Can targeted negative sampling repair presence without changing geometry or loss? | In development |

## Completed V1a.2 findings

### Negative-aware all-slice validation

The follow-up evaluation contains 3,644 validation slices, including 2,390
fully negative slices.

| Operating point | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|
| V1a.1 locked | 0.3097 | 0.7112 | 0.0464 |
| V1a.2-A locked | 0.7666 | **0.8556** | 0.7444 |
| V1a.2-B epoch 4 prompt-best | 0.3060 | 0.7534 | 0.0335 |
| V1a.2-B epoch 14 calibrated | **0.7725** | 0.8508 | **0.7598** |

The positive-slice V1a.2-B winner was not the best complete automatic operating
point. Epoch 14 is the best negative-aware V1a.2-B checkpoint, while V1a.2-A
remains slightly better in case-volume Dice.

### Fixed-scale geometry decomposition

All models were compared with ground-truth presence and expansion fixed at
`1.20`.

| Model | M0 auto box | Joint box gap | Center-size interaction |
|---|---:|---:|---:|
| V1a.1 | 0.8361 | 0.0347 | 0.0102 |
| V1a.2-A | 0.8406 | 0.0302 | 0.0128 |
| V1a.2-B | **0.8416** | **0.0292** | **0.0069** |

V1a.2-B is therefore the strongest geometry design. The remaining bottleneck
is presence detection and negative-slice suppression.

## Current development: V1a.3-A

V1a.3-A retains the complete V1a.2-B geometry and optimized loss. Its intended
change is limited to training-example sampling:

```text
V1a.2-B epoch-14 train predictions
    -> presence error audit
    -> hard-negative mining
    -> boundary-negative and unilateral-positive sampling
    -> V1a.3-A training from scratch
    -> validation-only calibration and negative-aware MedSAM
```

Default sampled-row categories are:

```text
positive_bilateral
positive_unilateral
negative_mined
negative_boundary
negative_near
negative_far
```

Training saves `best_presence.pt`, `best_geometry.pt`, and `best_joint.pt` as
prompt-network proxies. Final selection still requires validation-only
calibration and all-slice frozen-MedSAM evaluation.

See [V1a.3-A design and Colab runbook](v1/v1a3/README.md).

## Repository structure

```text
.
├── v0/                         # frozen-MedSAM prompt baselines
├── v0.5/                       # prompt robustness experiments
└── v1/
    ├── v1a1/                   # frozen baseline and test analysis
    ├── v1a2/                   # predicted-center size supervision
    ├── v1a2b/                  # decoded-box supervision and follow-up validation
    └── v1a3/                   # presence repair by targeted sampling
        ├── scripts/
        ├── v1a3/
        ├── DESIGN.md
        └── README.md
```

## Colab entry point for V1a.3-A

```bash
!git clone \
  -b v1a3-presence-repair \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

!bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a3/scripts/run_v1a3a_colab.sh
```

The helper expects the existing manifest, processed NPZ data, and V1a.2-B
epoch-14 checkpoint. Large assets remain in Google Drive and are not committed.

## Reproducibility

Git contains source code, locked configurations, compact comparison tables, and
summary documents. The repository intentionally excludes:

```text
model checkpoints (*.pt, *.pth)
MedSAM weights
raw KiTS23 volumes
processed NPZ slices
large per-slice prediction tables
full threshold-sweep tables
```

## Frozen-test status

The last frozen test evaluation remains V1a.1:

| Metric | Result |
|---|---:|
| Test cases | 27 |
| Positive slices | 1,473 |
| Slice-level mean Dice | 0.7628 |
| Case-level mean Dice | 0.7950 |
| Oracle-scaled mean Dice | 0.8713 |
| Automatic-to-oracle gap | 0.1085 |

V1a.2 and V1a.3 development have not used the frozen test split for selection.
