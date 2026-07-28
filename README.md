# MedSAM Automatic Prompt Generation on KiTS23

This repository contains a sequence of experiments for developing a lightweight, modular automatic spatial-prompt generator for frozen MedSAM on the KiTS23 kidney CT dataset.

## Experiment versions

| Version | Purpose | Status |
|---|---|---|
| [V0](v0/) | Frozen-MedSAM prompt baseline | Complete |
| [V0.5](v0.5/) | Bounding-box robustness analysis | Complete |
| [V1a.1](v1/v1a1/) | Dual-kidney 2.5D automatic prompt generator | Frozen test complete |

## Repository structure

```text
.
├── v0/
│   ├── baseline scripts
│   ├── reports
│   ├── selected cases
│   └── results
├── v0.5/
│   └── prompt_robustness/
└── v1/
    └── v1a1/
        ├── scripts/
        ├── v1a1/
        ├── splits/
        ├── docs/
        └── results/
```

## V1a.1 validation-locked configuration

```text
checkpoint      = epoch 6
weights         = EMA
prediction      = smooth
max_gap         = 0
left threshold  = 0.20
right threshold = 0.30
box expansion   = 1.20
```

## V1a.1 validation performance

| Metric | Result |
|---|---:|
| Slice-level mean Dice | 0.8122 |
| Slice-level median Dice | 0.9384 |
| Case-level mean Dice | 0.8233 |
| Overall oracle gap | 0.0587 |
| Case-level oracle gap | 0.0683 |

## V1a.1 frozen test performance

| Metric | Result |
|---|---:|
| Test cases | 27 |
| Positive slices | 1,473 |
| Slice-level mean Dice | 0.7628 |
| Slice-level median Dice | 0.9102 |
| Case-level mean Dice | 0.7950 |
| Oracle-scaled mean Dice | 0.8713 |
| Mean automatic-to-oracle gap | 0.1085 |

## Frozen-test gap decomposition

| Mode | Prompt source | Mean Dice |
|---|---|---:|
| A | Automatic presence + automatic box | 0.7628 |
| B | Ground-truth presence + automatic box | 0.7881 |
| C | Ground-truth presence + oracle box | 0.8713 |

The signed mean gap was associated approximately **23.3%** with presence decisions and **76.7%** with automatic-box localization/size. The failures were strongly long-tailed: the five largest-gap cases contributed about 78% of the positive oracle gap.

See [V1a.1 frozen test evaluation and error analysis](v1/v1a1/docs/FROZEN_TEST_EVALUATION_AND_ERROR_ANALYSIS.md).

## Current status

- Orientation audit: complete
- Canonical RAS conversion: complete
- Dual-kidney tracking: complete
- Dual-channel target generation: complete
- Small-sample overfit check: passed
- V1a.1 training: complete
- Validation calibration: complete
- Locked test evaluation: complete
- Test error analysis: complete
- Oracle-presence / automatic-box decomposition: complete
- Next: validation-only center-versus-size decomposition before defining V1a.2
