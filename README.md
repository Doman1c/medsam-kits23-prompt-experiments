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
| [V1a.3-A](v1/v1a3/) | Can targeted negative sampling repair presence without changing geometry or loss? | **Frozen test complete** |
| [V1a.3-B](v1/v1a3b/) | Can explicit case-level z-span decoding suppress residual presence islands without retraining? | **Validation in progress** |

## V1a.2 findings

### Negative-aware all-slice validation

The follow-up evaluation contains 3,644 validation slices, including 2,390
fully negative slices.

| Operating point | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|
| V1a.1 locked | 0.3097 | 0.7112 | 0.0464 |
| V1a.2-A locked | 0.7666 | **0.8556** | 0.7444 |
| V1a.2-B epoch 4 prompt-best | 0.3060 | 0.7534 | 0.0335 |
| V1a.2-B epoch 14 calibrated | **0.7725** | 0.8508 | **0.7598** |

V1a.2-B is the strongest geometry design, while V1a.2-A remains slightly
better in validation case-volume Dice.

### Fixed-scale geometry decomposition

All models were compared with ground-truth presence and expansion fixed at
`1.20`.

| Model | M0 auto box | Joint box gap | Center-size interaction |
|---|---:|---:|---:|
| V1a.1 | 0.8361 | 0.0347 | 0.0102 |
| V1a.2-A | 0.8406 | 0.0302 | 0.0128 |
| V1a.2-B | **0.8416** | **0.0292** | **0.0069** |

## Completed V1a.3-A study

V1a.3-A retained the V1a.2-B architecture and optimized loss and changed only
the training sampler. It used mined hard negatives, boundary negatives, near
negatives, limited far negatives, and unilateral-positive oversampling.

Validation selected epoch 18 with:

```text
EMA weights
smooth presence
max_gap = 2
left threshold = 0.005
right threshold = 0.005
expansion scale = 1.20
```

### Validation winner

| Metric | V1a.3-A epoch 18 |
|---|---:|
| Positive-slice Dice | 0.8067 |
| All-slice Dice | 0.7760 |
| Case-volume Dice | 0.8479 |
| Negative specificity | 0.7598 |

The fixed-presence M0 geometry score was `0.8416`, showing that targeted
sampling did not collapse V1a.2-B box geometry.

### Frozen test

The V1a.3-A protocol was locked before first test access and was not tuned
afterward.

| Metric | V1a.3-A frozen test |
|---|---:|
| Test cases | 27 |
| Total slices | 4,277 |
| Prompt positive-side recall | 0.9398 |
| Prompt side FPR | 0.2084 |
| Positive-slice MedSAM Dice | 0.7239 |
| Positive-slice case-mean Dice | 0.7627 |
| All-slice Dice | 0.7475 |
| Case-volume Dice | 0.7996 |
| Negative specificity | 0.7603 |

V1a.3-A improved the controlled validation picture for negative-slice
suppression while preserving geometry, but its frozen-test positive-slice Dice
did not exceed the V1a.1 frozen baseline (`0.7628`). It is therefore retained as
a completed negative-sampling study rather than a new overall system winner.

See [V1a.3-A protocol and results](v1/v1a3/) and the
[compact result bundle](v1/v1a3/results/).

## Current development: V1a.3-B

V1a.3-B freezes the V1a.3-A epoch-18 EMA network and changes only the case-level
presence decoder. The first controlled comparison is:

```text
B0 locked V1a.3-A threshold + smoothing + gap fill
B1 retain one dominant threshold-derived segment per kidney side
B2 choose one optimal contiguous kidney span with bounded internal low-confidence gap
```

The branch includes a fresh-Colab environment recovery script, a topology audit,
a recall-constrained decoder sweep, prompt-level shortlist evaluation, and an
optional negative-aware MedSAM stage. Development is restricted to the 20-case
validation split; no new V1a.3-B test access is permitted before validation
selection is complete.

See [V1a.3-B design and runbook](v1/v1a3b/).

## Repository structure

```text
.
├── v0/                         # frozen-MedSAM prompt baselines
├── v0.5/                       # prompt robustness experiments
└── v1/
    ├── v1a1/                   # frozen baseline and test analysis
    ├── v1a2/                   # predicted-center size supervision
    ├── v1a2b/                  # decoded-box supervision and follow-up validation
    ├── v1a3/                   # completed targeted-sampling presence repair
    └── v1a3b/                  # case-level z-span presence decoding
        ├── scripts/
        ├── v1a3b/
        ├── DESIGN.md
        └── README.md
```

## Reproducibility

Git contains source code, locked configurations, compact comparison tables, and
summary documents. The repository intentionally excludes:

```text
notebooks used only for Colab recovery
model checkpoints (*.pt, *.pth)
MedSAM weights
raw KiTS23 volumes
processed NPZ slices
large per-slice prediction tables
full threshold-sweep tables
```

Large immutable artifacts remain outside Git, while their locked SHA-256 values
are recorded in the compact result files.

## Frozen-test comparison

| Model | Positive-slice Dice | Positive case-mean Dice |
|---|---:|---:|
| V1a.1 | **0.7628** | **0.7950** |
| V1a.3-A epoch 18 | 0.7239 | 0.7627 |

No V1a.3-A parameter was changed after the frozen test was opened. V1a.3-B is a
new validation-only experiment and does not tune against that frozen result.
