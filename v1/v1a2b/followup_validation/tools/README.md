# V1a.2 follow-up validation tools

These scripts complete the three remaining validation-only tests without
retraining and without reading the frozen test split.

## Test 1 — Negative-aware all-slice evaluation

Evaluates all 3,644 validation slices, including fully negative slices.

Compared candidates:

- V1a.1 locked configuration
- V1a.2-A locked configuration
- V1a.2-B prompt-best
- V1a.2-B FPR-best
- V1a.2-B continuous-best
- V1a.2-B target-recall, when available

Metrics include:

- positive-slice Dice
- all-slice Dice
- case-volume Dice
- negative-slice specificity
- false-positive mask rate
- false-positive mask area fraction
- prompt rate on negative slices

The MedSAM image embedding is shared across candidates for each slice.

## Test 2 — Fixed expansion 1.20 decomposition

Runs M0/M1/M2/M3 for:

- V1a.1
- V1a.2-A
- V1a.2-B

All use GT presence and expansion 1.20.

## Test 3 — V1a.2-B checkpoint Pareto

Recalibrates and evaluates epochs 4, 5, 11, and 14. Each checkpoint receives an independent validation-only threshold/smoothing search, candidate MedSAM sweep, and negative-aware evaluation.

The final descriptive ranking is case-volume Dice, then all-slice mean Dice, then lower negative false-positive mask rate.
