# V1a.2 follow-up validation

This directory contains three validation-only checks completed after the initial
V1a.2-B analysis. The frozen test split was not used.

## Purpose

The first V1a.2-B operating point improved positive-slice Dice and box geometry,
but it also produced many false prompts on slices without kidneys. The follow-up
therefore evaluates the complete automatic system, repeats geometry analysis at
a common expansion scale, and compares multiple checkpoints after independent
validation calibration.

## Validation set

```text
validation cases        = 20
all validation slices   = 3,644
positive kidney slices  = 1,254
fully negative slices   = 2,390
```

## 1. Negative-aware all-slice evaluation

| Candidate | All-slice Dice | Case-volume Dice | Negative specificity | Positive-slice Dice |
|---|---:|---:|---:|---:|
| V1a.1 locked | 0.3097 | 0.7112 | 0.0464 | 0.8115 |
| V1a.2-A locked | **0.7666** | **0.8556** | **0.7444** | 0.8089 |
| V1a.2-B FPR-best | 0.6503 | 0.8238 | 0.5762 | 0.7916 |
| V1a.2-B continuous-best | 0.6525 | 0.8162 | 0.5741 | 0.8019 |
| V1a.2-B target-recall | 0.5403 | 0.7907 | 0.4050 | 0.7982 |
| V1a.2-B prompt-best | 0.3060 | 0.7534 | 0.0335 | **0.8256** |

The positive-slice V1a.2-B winner is not the strongest automatic operating
point. V1a.2-A locked has the highest case-volume Dice in this comparison.

## 2. Fixed expansion 1.20 decomposition

All models use ground-truth presence and expansion `1.20`.

| Model | M0 | M1 | M2 | M3 | Joint gap | Interaction |
|---|---:|---:|---:|---:|---:|---:|
| V1a.1 | 0.8361 | 0.8510 | 0.8457 | 0.8708 | 0.0347 | 0.0102 |
| V1a.2-A | 0.8406 | 0.8537 | 0.8450 | 0.8708 | 0.0302 | 0.0128 |
| V1a.2-B | **0.8416** | **0.8554** | **0.8501** | 0.8708 | **0.0292** | **0.0069** |

V1a.2-B is the strongest geometry model under the controlled comparison.

## 3. V1a.2-B checkpoint Pareto

Each checkpoint was independently calibrated on validation data.

| Epoch | Calibration | All-slice Dice | Case-volume Dice | Negative specificity | Positive-slice Dice |
|---:|---|---:|---:|---:|---:|
| 4 | smooth, gap 0, 0.10/0.20, scale 1.25 | 0.3060 | 0.7534 | 0.0335 | **0.8262** |
| 5 | raw, 0.05/0.075, scale 1.25 | 0.3996 | 0.7677 | 0.1791 | 0.8206 |
| 11 | smooth, gap 2, 0.01/0.01, scale 1.15 | 0.7415 | 0.8485 | 0.7121 | 0.7981 |
| 14 | smooth, gap 2, 0.01/0.01, scale 1.15 | **0.7725** | **0.8508** | **0.7598** | 0.7972 |

Epoch 14 is the preferred V1a.2-B checkpoint for negative-aware automatic use.

## Final decision

- Retain V1a.2-B decoded-box supervision as the leading geometry design.
- Use epoch 14 instead of epoch 4 for automatic V1a.2-B evaluation.
- Keep V1a.2-A locked as the best evaluated case-volume operating point.
- Focus the next stage on presence detection and negative-slice suppression.

## Contents

```text
followup_validation/
├── tools/      # evaluators and shell entry points
└── results/    # compact CSV/JSON summaries and locked configurations
```

Large per-slice tables, checkpoints, processed NPZ files, and MedSAM weights are
stored outside Git.
