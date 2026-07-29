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
or loss variants.

## Experiment versions

| Version | Main question | Status |
|---|---|---|
| [V0](v0/) | Which MedSAM prompt combinations work best? | Complete |
| [V0.5](v0.5/) | How robust is MedSAM to box and mask degradation? | Complete |
| [V1a.1](v1/v1a1/) | Can a 2.5D dual-kidney network generate automatic boxes? | Frozen test complete |
| [V1a.2-A](v1/v1a2/) | Does predicted-center size supervision repair the train-inference mismatch? | Validation complete |
| [V1a.2-B](v1/v1a2b/) | Does decoded-corner supervision reduce joint center-size error? | Validation complete |
| [Follow-up validation](v1/v1a2b/followup_validation/) | Which model survives fully negative slices and checkpoint-Pareto analysis? | Complete |

## Current validation findings

### Positive-slice MedSAM evaluation

| Model | Slice Dice | Case Dice | Notes |
|---|---:|---:|---|
| V1a.1 | 0.8122 | 0.8233 | Frozen baseline |
| V1a.2-A | 0.8096 | 0.8261 | Better geometry, no slice-Dice gain |
| V1a.2-B epoch 4 | **0.8262** | **0.8419** | Best positive-slice operating point |

### Negative-aware all-slice evaluation

The follow-up evaluation contains 3,644 validation slices, including 2,390
fully negative slices.

| Operating point | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|
| V1a.1 locked | 0.3097 | 0.7112 | 0.0464 |
| V1a.2-A locked | 0.7666 | **0.8556** | 0.7444 |
| V1a.2-B epoch 4 prompt-best | 0.3060 | 0.7534 | 0.0335 |
| V1a.2-B epoch 14 calibrated | **0.7725** | 0.8508 | **0.7598** |

The positive-slice V1a.2-B winner was not the best fully automatic operating
point. Recalibrating later checkpoints changed the conclusion: epoch 14 is the
best negative-aware V1a.2-B checkpoint, while V1a.2-A remains slightly better
in case-volume Dice.

### Fixed-scale geometry decomposition

All models were compared with ground-truth presence and expansion fixed at
`1.20`.

| Model | M0 auto box | Joint box gap | Center-size interaction |
|---|---:|---:|---:|
| V1a.1 | 0.8361 | 0.0347 | 0.0102 |
| V1a.2-A | 0.8406 | 0.0302 | 0.0128 |
| V1a.2-B | **0.8416** | **0.0292** | **0.0069** |

V1a.2-B is therefore the strongest geometry design. The remaining bottleneck
is presence detection and negative-slice suppression, not additional box-loss
complexity.

## Current decision

- Retain decoded-box supervision from V1a.2-B.
- Use epoch 14 rather than epoch 4 for negative-aware V1a.2-B evaluation.
- Keep V1a.2-A locked as the best evaluated case-volume operating point.
- Do not freeze a final V1a.2 automatic model yet.
- Focus the next development stage on presence calibration, negative-slice
  suppression, and onset/offset behavior.

## Repository structure

```text
.
├── v0/                         # frozen-MedSAM prompt baselines
├── v0.5/                       # prompt robustness experiments
└── v1/
    ├── v1a1/                   # frozen baseline and test analysis
    ├── v1a2/                   # V1a.2-A predicted-center size supervision
    └── v1a2b/                  # V1a.2-B decoded-box supervision
        ├── scripts/            # training, prediction, and gates
        ├── v1a2b/              # model/runtime/loss package
        ├── docs/               # design and validation status
        └── followup_validation/
            ├── tools/          # negative-aware and Pareto evaluators
            └── results/        # compact validation summaries
```

## Reproducibility

Git contains source code, locked configurations, compact comparison tables, and
summary documents. Large artifacts are intentionally excluded:

```text
model checkpoints (*.pt, *.pth)
MedSAM weights
raw KiTS23 volumes
processed NPZ slices
large per-slice prediction tables
full threshold-sweep tables
```

Those artifacts are stored outside the repository, typically in Google Drive.
See:

- [V1a.2-B validation status](v1/v1a2b/docs/V1A2_VALIDATION_STATUS.md)
- [Follow-up validation protocol and results](v1/v1a2b/followup_validation/README.md)
- [Fresh-Colab asset checklist](v1/v1a2b/docs/RESUME_FOLLOWUP_VALIDATION.md)

## Frozen V1a.1 test result

The last frozen test evaluation remains V1a.1:

| Metric | Result |
|---|---:|
| Test cases | 27 |
| Positive slices | 1,473 |
| Slice-level mean Dice | 0.7628 |
| Case-level mean Dice | 0.7950 |
| Oracle-scaled mean Dice | 0.8713 |
| Automatic-to-oracle gap | 0.1085 |

V1a.2-A and V1a.2-B have not been selected on the frozen test split.
