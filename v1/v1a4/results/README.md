# V1a.4 compact results

This directory stores compact, reviewable V1a.4 result summaries only. Large
prediction tables, model checkpoints, NPZ data, MedSAM weights, and notebooks are
kept out of Git.

## Stage 1

`stage1_summary.json` records the validation-only recovery audit and sweep
outcome. The key finding is that local-only recall recovery is insufficient:

- frozen B1 recall/FPR: `0.9502 / 0.1193`;
- 107 false-negative side-slices;
- 83 misses before the frozen seed and 85 on the left;
- only 47/107 misses are within 16 slices of the seed;
- one left case-side contributes 76 misses extending up to 76 slices before the
  selected dominant span;
- best local hysteresis reaches recall `0.9567` at FPR `0.1341` but does not meet
  the locked V1a.4 gate;
- the original local-satellite grid (distance <=16) does not improve recall.

`stage1_best_points.csv` provides the frozen B1 and best local-hysteresis
operating points in compact tabular form.

Stage 1b therefore tests one strongly supported remote satellite segment, alone
and in a controlled hybrid with the fixed Stage-1 local hysteresis rule. The
original validation gate is unchanged, and no MedSAM/test access is allowed until
that gate is satisfied.
