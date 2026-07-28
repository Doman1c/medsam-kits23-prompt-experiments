# V1a.1 Validation Center-versus-Size Decomposition

## Purpose

This validation-only diagnostic separates automatic-box error into center, size, and center-size interaction components before defining V1a.2. The frozen test set is not used.

All four modes use ground-truth left/right presence and the validation-locked V1a.1 configuration:

```text
checkpoint      = epoch 6
weights         = EMA
prediction      = smooth
max_gap         = 0
left threshold  = 0.20
right threshold = 0.30
box expansion   = 1.20
target label    = 1
```

## Modes

| Mode | Presence | Center | Size |
|---|---|---|---|
| M0 | Ground truth | Automatic smooth | Automatic smooth |
| M1 | Ground truth | Oracle | Automatic smooth |
| M2 | Ground truth | Automatic smooth | Oracle |
| M3 | Ground truth | Oracle | Oracle |

The modes are compared with frozen MedSAM on all positive validation slices.

## Data and runtime

| Item | Value |
|---|---:|
| Validation cases | 20 |
| Positive validation slices | 1,253 |
| Mode-result rows | 5,012 |
| Device | CUDA |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Runtime | 51.37 seconds |
| Peak CUDA memory | 1,591 MiB |

## Overall results

| Mode | Mean Dice | Median Dice | Mean IoU |
|---|---:|---:|---:|
| M0: automatic center + automatic size | 0.8361 | 0.9436 | 0.7784 |
| M1: oracle center + automatic size | 0.8510 | 0.9461 | 0.7924 |
| M2: automatic center + oracle size | 0.8457 | 0.9479 | 0.7916 |
| M3: oracle center + oracle size | 0.8708 | 0.9487 | 0.8153 |

The mean gains are:

```text
oracle-center gain (M1 - M0) = 0.0149
oracle-size gain   (M2 - M0) = 0.0096
joint box gain     (M3 - M0) = 0.0347
interaction                     = 0.0102
```

The signed standalone center gain accounts for about 42.9% of the joint gap, the signed standalone size gain for about 27.6%, and the remaining difference is associated with center-size interaction. Center and size effects are therefore not assumed additive.

## Relative-area analysis

| Area bin | M0 | M1 | M2 | M3 | Center gain | Size gain | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-10% | 0.3694 | 0.4298 | 0.3958 | 0.4987 | 0.0604 | 0.0264 | 0.0425 |
| 10-50% | 0.9058 | 0.9174 | 0.9248 | 0.9372 | 0.0116 | 0.0190 | 0.0008 |
| 50-90% | 0.9161 | 0.9242 | 0.9171 | 0.9335 | 0.0081 | 0.0010 | 0.0084 |
| 90-100% | 0.9341 | 0.9318 | 0.9336 | 0.9342 | -0.0023 | -0.0005 | 0.0030 |

Main observations:

- The 0-10% stratum has the largest joint gap. Center error is the larger standalone component, and interaction is also substantial.
- The 10-50% stratum is the clearest size-sensitive group.
- The 50-90% stratum is mainly affected by center error and center-size interaction rather than size alone.
- The 90-100% stratum is effectively saturated; automatic boxes and oracle boxes produce nearly identical mean Dice.

## Case-level long-tail examples

| Case | M0 | M1 | M2 | M3 | Main pattern |
|---|---:|---:|---:|---:|---|
| case_00431 | 0.7783 | 0.9000 | 0.8209 | 0.9317 | Center-dominant |
| case_00107 | 0.6340 | 0.6750 | 0.6241 | 0.7869 | Strong interaction |
| case_00067 | 0.4719 | 0.5086 | 0.5159 | 0.5746 | Center and size both matter |
| case_00418 | 0.7284 | 0.7454 | 0.7260 | 0.8044 | Interaction-dominant |
| case_00064 | 0.8747 | 0.8820 | 0.9121 | 0.9265 | Size-dominant |

## Interpretation for V1a.2

V1a.1 supervises the size map at the ground-truth center during training, while inference reads size at the predicted heatmap peak. A center error can therefore change both the predicted center and the spatial location from which width and height are read. The observed interaction is consistent with this train-inference mismatch.

The primary V1a.2 hypothesis should be:

> Inference-consistent center-conditioned size training, combined with direct decoded-box supervision, can reduce center-size interaction and long-tailed box failures.

A controlled validation ablation should compare:

1. V1a.1 baseline.
2. Predicted-center or soft-center size-map sampling.
3. Predicted-center sampling plus decoded-box IoU/GIoU supervision.
4. Optional small-area center weighting only after the first two changes are evaluated.

Presence specificity remains a secondary objective. The frozen test set must not be revisited for selecting loss weights, schedules, expansion scales, checkpoints, or architecture variants.

## Archived artifacts

- `scripts/evaluate_validation_center_size_decomposition.py`
- `results/validation_center_size_decomposition/center_size_decomposition_summary.json`

Large per-slice CSV files and model/data artifacts remain on Drive. Their SHA256 values are recorded in the summary JSON.
