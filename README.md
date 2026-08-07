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

Development follows controlled validation-only ablations. The original test
split is not used to select new V1a.4 decoder parameters. Because that split was
already exposed in earlier frozen evaluations, any later use is explicitly
reported as a locked secondary evaluation rather than a pristine first-look
estimate.

## Experiment versions

| Version | Main question | Status |
|---|---|---|
| [V0](v0/) | Which MedSAM prompt combinations work best? | Complete |
| [V0.5](v0.5/) | How robust is MedSAM to prompt degradation? | Complete |
| [V1a.1](v1/v1a1/) | Can a 2.5D dual-kidney network generate automatic boxes? | Frozen test complete |
| [V1a.2-A](v1/v1a2/) | Does predicted-center size supervision repair train-inference mismatch? | Validation complete |
| [V1a.2-B](v1/v1a2b/) | Does decoded-corner supervision reduce joint center-size error? | Validation complete |
| [V1a.2 follow-up](v1/v1a2b/followup_validation/) | Which model survives fully negative slices and checkpoint-Pareto analysis? | Complete |
| [V1a.3-A](v1/v1a3/) | Can targeted negative sampling repair presence without changing geometry or loss? | Frozen test complete |
| [V1a.3-B](v1/v1a3b/) | Can explicit case-level z-span decoding suppress residual presence islands without retraining? | **Complete** |
| [V1a.4](v1/v1a4/) | Can local recovery restore recall without giving back V1a.3-B false-positive suppression? | **Validation in progress** |

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
the training sampler. Validation selected epoch 18 EMA with smooth presence,
`max_gap=2`, left/right threshold `0.005/0.005`, and expansion `1.20`.

| Metric | Validation | Frozen test |
|---|---:|---:|
| Positive-slice Dice | 0.8067 | 0.7239 |
| All-slice Dice | 0.7760 | 0.7475 |
| Case-volume Dice | 0.8479 | 0.7996 |
| Negative specificity | 0.7598 | 0.7603 |

The fixed-presence M0 geometry score was `0.8416`, showing that targeted
sampling did not collapse V1a.2-B box geometry.

## Completed V1a.3-B study

V1a.3-B froze the V1a.3-A epoch-18 EMA network and changed only case-level
presence decoding. Validation selected a simple dominant single-span decoder:

```text
left threshold      = 0.0035
right threshold     = 0.0400
baseline max gap    = 4
boundary pad        = 1
expansion            = 1.20
```

### Validation

| Metric | V1a.3-A / B0 | V1a.3-B / B1 |
|---|---:|---:|
| All-slice Dice | 0.7760 | **0.8356** |
| Positive-slice Dice | 0.8067 | **0.8102** |
| Negative specificity | 0.7598 | **0.8490** |
| Case-volume Dice | 0.8479 | **0.8548** |

### Locked secondary evaluation

The original 27-case split had already been exposed before V1a.3-B, so these
numbers are reported only as a locked secondary comparison.

| Metric | B0 | Frozen B1 |
|---|---:|---:|
| All-slice Dice | 0.7475 | **0.8009** |
| Positive-slice Dice | **0.7234** | 0.6827 |
| Positive case-mean Dice | **0.7627** | 0.7587 |
| Positive-slice detection recall | **0.9715** | 0.8636 |
| Negative specificity | 0.7603 | **0.8630** |
| Case-volume Dice | 0.7996 | **0.8016** |

The z-span prior generalizes strongly for false-positive suppression and
all-slice robustness, but the hard single-span constraint can over-prune true
positive slices. V1a.3-B is therefore a conservative operating point rather
than an unconditional replacement for B0.

## Current development: V1a.4

V1a.4 starts from the frozen V1a.3-B B1 configuration and asks whether recall
can be recovered without returning to the detached false-positive islands that
V1a.3-B removed.

The first controlled validation-only comparison is:

```text
R0 frozen V1a.3-B B1
R1 hysteresis boundary recovery
R2 retain at most one nearby supported satellite segment
```

Stage 1 first audits where frozen-B1 false-negative slices lie relative to the
predicted dominant span, then sweeps R1/R2 under a locked validation gate:

```text
overall recall >= 0.96
left recall    >= 0.94
right recall   >= 0.97
FPR             <= 0.15
```

No MedSAM or test split is used in the first V1a.4 stage. See
[V1a.4 design and runbook](v1/v1a4/).

## Repository structure

```text
.
├── v0/
├── v0.5/
└── v1/
    ├── v1a1/
    ├── v1a2/
    ├── v1a2b/
    ├── v1a3/
    ├── v1a3b/
    └── v1a4/
        ├── scripts/
        ├── v1a4/
        ├── BASELINE_LOCK.json
        ├── DESIGN.md
        └── README.md
```

## Reproducibility

Git contains source code, locked configurations, compact comparison tables, and
summary documents. The repository intentionally excludes notebooks used only
for Colab recovery, model checkpoints, MedSAM weights, raw KiTS23 volumes,
processed NPZ slices, and large per-slice prediction tables. Large immutable
artifacts remain outside Git, while locked hashes are recorded in compact
configuration/result files.
