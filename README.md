# MedSAM Automatic Prompt Generation on KiTS23

This repository contains the reproducible code and experiment records for automatic spatial prompt generation for MedSAM on the KiTS23 kidney CT dataset.

The project goal is to replace manual bounding-box prompting with an automatic lightweight prompt generator that predicts reliable spatial prompts for MedSAM.

---

## Research Pipeline

```
KiTS23 CT slices
        |
        v
Canonical orientation normalization (RAS)
        |
        v
Dual kidney tracking
        |
        v
Left / Right prompt generator
        |
        v
Automatic bounding boxes
        |
        v
MedSAM frozen segmentation
        |
        v
Dice evaluation
```

---

# Versions

## V0: Prompt Ablation Baseline

Purpose:
- Validate whether automatically generated spatial prompts can drive MedSAM.
- Compare box, mask-derived box, positive/negative prompts.

Main files:

```
run_v0_experiment.py
prepare_kits23_2d.py
```

---

# V1a.1 Dual Kidney Prompt Generator

## Motivation

V0 used largest connected component extraction, which can introduce:

- left/right kidney identity switching
- unstable component selection across slices
- poor temporal consistency

V1a.1 introduces anatomical consistency through dual kidney tracking.

---

## Completed Components

### 1. Orientation Validation

All 147 KiTS23 cases were checked.

Results:

- Original orientation: IPL
- Canonical orientation: RAS
- Shape mismatch: 0
- Affine mismatch: 0

All cases successfully converted to canonical RAS orientation.

---

### 2. Dual Kidney Tracking

Implemented:

- left kidney tracking
- right kidney tracking
- slice-to-slice anatomical consistency
- side identity preservation

Validation:

- left crossings: 0
- right crossings: 0

---

### 3. Dual Channel Prompt Generation

Generated targets:

- kidney presence
- center heatmap
- local size prediction
- center offset
- bounding box parameters
- relative area information

Both left and right kidneys are modeled independently.

---

### 4. Overfit Verification

Before full training, a small-sample overfit test was performed.

Passed:

- positive recall
- center localization
- box recovery
- prompt generation

This verifies the training pipeline and loss implementation.

---

# V1a.1 Training

Configuration:

```
input: 2.5D CT slices
base channels: 24
optimizer: AdamW
learning rate: 3e-4
EMA enabled
```

Training stopped using early stopping.

Best checkpoint selected:

```
epoch6
```

---

# Post Training Calibration

Calibration searched:

- checkpoint selection
- probability threshold
- z-axis smoothing
- box expansion

Final validation configuration:

```
checkpoint: epoch6
prediction variant: smooth
max_gap: 0
left threshold: 0.20
right threshold: 0.30
box expansion: 1.20
```

Validation performance:

| Metric | Value |
|---|---:|
| Slice Dice mean | 0.812 |
| Slice Dice median | 0.938 |
| Case Dice mean | 0.823 |
| Oracle gap | 0.059 |

Relative area analysis:

| Kidney area | Dice |
|---|---:|
| 0-10% | 0.277 |
| 10-50% | 0.870 |
| 50-90% | 0.916 |
| 90-100% | 0.934 |

The remaining challenge is extremely small kidney regions near slice boundaries.

---

# Repository Structure

```
.
├── v1a_prompt_generator/
├── v1a_complete_pipeline/
├── v1a1_dual_kidney_pipeline/
├── calibration/
├── results/
└── README.md
```

---

# Data and Model Weights

Large files are intentionally excluded:

- KiTS23 raw CT volumes
- processed NPZ dataset
- MedSAM checkpoint
- training checkpoints

Expected paths in Colab:

```
/content/drive/MyDrive/medsam-kits23
/content/drive/MyDrive/medsam_vit_b.pth
```

---

# Reproducibility

Recommended workflow:

1. Mount Google Drive in Colab.
2. Download this repository.
3. Restore KiTS23 archive and MedSAM checkpoint.
4. Run preprocessing.
5. Build dual kidney manifest.
6. Train prompt generator.
7. Run calibration.
8. Evaluate MedSAM segmentation.

---

# Current Status

| Stage | Status |
|-|-|
| V0 baseline | Completed |
| Oracle area analysis | Completed |
| Orientation validation | Completed |
| Dual kidney tracking | Completed |
| Dual prompt labels | Completed |
| Overfit validation | Completed |
| V1a.1 training | Completed |
| Calibration | Completed |
| Final test evaluation | Pending |

---

# Future Work

- final test-set evaluation
- improve small-area kidney prompt generation
- uncertainty-aware multi-prompt generation
- language-guided prompt generation
