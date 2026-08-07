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

## Stage 1b

`stage1b_summary.json` records the widened validation-only satellite study. The
360-config sweep produced 40 feasible points under the unchanged gate: 20
remote-satellite and 20 hybrid configurations.

Pre-MedSAM finalists:

| Candidate | Recall | Left recall | Right recall | FPR |
|---|---:|---:|---:|---:|
| Frozen B1 | 0.9502 | 0.9275 | 0.9775 | 0.1193 |
| Fixed local hysteresis | 0.9567 | 0.9318 | 0.9867 | 0.1341 |
| R2b remote satellite | 0.9619 | 0.9488 | 0.9775 | 0.1193 |
| R3 hybrid | 0.9684 | 0.9531 | 0.9867 | 0.1341 |

The recovery appears only when the satellite distance reaches 64 slices. R2b
crosses the locked recall gate without increasing frozen-B1 FPR on validation.
`STAGE2_LOCK.json` freezes exactly four candidates before MedSAM evaluation.

## Stage 2 — negative-aware frozen MedSAM

The Stage-2 review bundle passed all listed size/SHA-256 checks and contains no
test results, model weights, or NPZ data.

The predeclared ranking rule is:

```text
case-volume Dice
-> all-slice mean Dice
-> lower negative false-positive mask rate
```

Results:

| Candidate | All-slice Dice | Positive Dice | Detection recall | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|---:|
| **Frozen B1** | 0.83561 | 0.81018 | 0.93381 | **0.84895** | **0.85475** |
| Fixed local hysteresis | 0.82979 | 0.81400 | 0.93541 | 0.83808 | 0.85447 |
| R2b remote satellite | **0.83579** | 0.81070 | **0.95136** | **0.84895** | 0.85393 |
| R3 hybrid | 0.82997 | **0.81453** | **0.95295** | 0.83808 | 0.85365 |

R2b is the cleanest presence recovery: it increases detection recall while
preserving negative specificity and slightly raises all-slice/positive-slice
Dice. However, it does **not** improve the primary case-volume metric.

The per-slice diagnostic explains why. Relative to frozen B1, R2b adds prompts on
22 combined GT-positive slices and no GT-negative slices. All 22 belong to
`case_00067`. Those recovered slices add only 650 TP pixels but 19,374 FP pixels;
their mean Dice is about `0.0298` and median Dice is `0.0`. Consequently,
`case_00067` case-volume Dice drops by about `0.01635`, lowering mean case-volume
Dice from `0.85475` to `0.85393`.

This shows that the remaining bottleneck is no longer only presence decoding.
The network can identify additional remote positive slices, but the frozen box
geometry on those low-confidence/recovered slices is not reliable enough for
MedSAM.

## Final V1a.4 decision

`FINAL_DECISION.json` freezes the validation conclusion:

- no V1a.4 recovery decoder replaces frozen V1a.3-B B1 under the predeclared
  Stage-2 ranking rule;
- no V1a.4 secondary-test run is warranted;
- no post-Stage-2 decoder tuning is allowed;
- future development should target prompt/box geometry on recovered or
  low-confidence slices rather than widening case-level presence post-processing.

Compact machine-readable files:

- `stage1_summary.json`
- `stage1_best_points.csv`
- `stage1b_summary.json`
- `stage1b_finalists.csv`
- `stage2_summary.json`
- `stage2_comparison.csv`
