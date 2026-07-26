# MedSAM Automatic Prompt Generation on KiTS23

This repository contains a sequence of experiments for developing a lightweight, modular automatic spatial-prompt generator for frozen MedSAM on the KiTS23 kidney CT dataset.

## Research pipeline

```text
2.5D CT slices
    ↓
Automatic left/right kidney localization
    ↓
Presence + center heatmaps + local box regression
    ↓
Calibrated bounding boxes
    ↓
Frozen MedSAM
    ↓
Kidney segmentation
```

## Experiment versions

| Version | Purpose | Status |
|---|---|---|
| [V0](v0/) | Frozen-MedSAM prompt baseline | Complete |
| [V0.5](v0.5/) | Bounding-box robustness analysis | Complete |
| [V1a.1](v1/v1a1/) | Dual-kidney 2.5D automatic prompt generator | Validation complete |

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
prediction      = smooth
max_gap         = 0
left threshold  = 0.20
right threshold = 0.30
box expansion   = 1.20
```

## Validation performance

| Metric | Result |
|---|---:|
| Slice-level mean Dice | 0.8122 |
| Slice-level median Dice | 0.9384 |
| Case-level mean Dice | 0.8233 |
| Overall oracle gap | 0.0587 |
| Case-level oracle gap | 0.0683 |

### Performance by relative kidney area

| Relative area | Dice |
|---|---:|
| 0–10% | 0.2772 |
| 10–50% | 0.8697 |
| 50–90% | 0.9156 |
| 90–100% | 0.9341 |

The main remaining limitation is segmentation of extremely small kidney components near the superior and inferior edges of a CT volume.

## Excluded assets

The following are intentionally excluded from Git:

- KiTS23 NIfTI volumes
- processed NPZ slices
- MedSAM weights
- trained checkpoints
- large per-slice prediction tables

## Current status

- Orientation audit: complete
- Canonical RAS conversion: complete
- Dual-kidney tracking: complete
- Dual-channel target generation: complete
- Small-sample overfit check: passed
- V1a.1 training: complete
- Validation calibration: complete
- Locked test evaluation: pending
