# V1a.2 follow-up validation

This directory archives the three validation-only checks completed after the initial V1a.2-B analysis. The frozen test split was not used.

## Completed checks

1. Negative-aware MedSAM evaluation on all 3,644 validation slices.
2. Center-size decomposition with expansion fixed to 1.20 for V1a.1, V1a.2-A, and V1a.2-B.
3. Independent validation calibration and checkpoint Pareto analysis for V1a.2-B epochs 4, 5, 11, and 14.

## Main result

V1a.2-B remains the strongest geometry model, but V1a.2-A has the best evaluated locked negative-aware case-volume Dice. Presence behavior remains the limiting factor for the fully automatic system.

Exact scripts and compact results are stored in the sibling `tools/` and `results/` directories. Large per-slice details, checkpoints, processed NPZ files, MedSAM weights, and full threshold-sweep tables remain on Google Drive.
