# V1 — Learned Automatic Prompt Generation

V1 contains learned automatic spatial-prompt generation methods.

## Current implementation

### [V1a.1 Dual-Kidney Prompt Generator](v1a1/)

V1a.1 uses 2.5D CT input and predicts left/right kidney presence, center heatmaps, center-local box sizes, offsets, and calibrated bounding boxes. The two boxes are passed separately to frozen MedSAM and the resulting masks are combined.

**Status:** training, validation calibration, one-time frozen test evaluation, and test error analysis are complete.

## Frozen test result

| Metric | Result |
|---|---:|
| Test cases | 27 |
| Positive slices | 1,473 |
| Automatic slice mean Dice | 0.7628 |
| Automatic slice median Dice | 0.9102 |
| Automatic case mean Dice | 0.7950 |
| Oracle-scaled slice mean Dice | 0.8713 |
| Mean oracle gap | 0.1085 |

## Oracle-gap decomposition

| Mode | Mean Dice |
|---|---:|
| Automatic presence + automatic box | 0.7628 |
| Oracle presence + automatic box | 0.7881 |
| Oracle presence + oracle box | 0.8713 |

The diagnostic decomposition attributes about **23.3%** of the signed mean oracle gap to presence decisions and **76.7%** to automatic-box localization/size. This indicates that V1a.2 should not be defined as a presence-only modification.

See [Frozen Test Evaluation and Error Analysis](v1a1/docs/FROZEN_TEST_EVALUATION_AND_ERROR_ANALYSIS.md).

## Next development gate

V1a.2 has not started. The next step is a **validation-only center-versus-size box decomposition**, followed by a pre-specified V1a.2 design. The frozen test set must not be reused to choose thresholds, losses, expansion scales, checkpoints, or model variants.
