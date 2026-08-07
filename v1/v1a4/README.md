# V1a.4 — Recall recovery after hard single-span pruning

V1a.4 is a validation-only post-processing study built on the frozen V1a.3-B
single-span decoder. It does not retrain the prompt network and does not change
box geometry, MedSAM, target label, or expansion.

## Frozen upstream

```text
V1a.3-B merge       = 36c260096f733fc54e4db0819cc04527d021eff6
checkpoint          = V1a.3-A epoch 18 EMA
checkpoint SHA256   = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2

frozen B1 seed:
left threshold      = 0.0035
right threshold     = 0.0400
baseline max gap    = 4
boundary pad        = 1
expansion scale     = 1.20
```

The exact lock is stored in `BASELINE_LOCK.json`.

## Locked validation gate

All V1a.4 decoder development uses only the 20-case / 3,644-row validation
split.

```text
overall recall >= 0.96
left recall    >= 0.94
right recall   >= 0.97
FPR             <= 0.15
```

No MedSAM is run before this presence-level gate is passed. The already-exposed
secondary test is never used for V1a.4 parameter selection.

## Stage 1 — local recovery

Stage 1 tested:

```text
R0 frozen B1
R1 bounded lower-threshold hysteresis around the dominant span
R2 one nearby threshold-derived satellite segment
```

Frozen B1 validation presence:

| Metric | Frozen B1 |
|---|---:|
| Overall recall | 0.9502 |
| Left recall | 0.9275 |
| Right recall | 0.9775 |
| FPR | 0.1193 |
| Specificity | 0.8807 |

The audit found 107 false-negative side-slices. Eighty-three occur before the
frozen seed, 85 are left-side misses, and only 47/107 lie within 16 slices of the
seed. One validation case-side (`case_00067`, left) accounts for 76 misses and
extends up to 76 slices before the selected dominant span.

The best local hysteresis point under `FPR <= 0.15` was:

```text
left recovery threshold  = 0.0005
right recovery threshold = 0.0025
max extension            = 6
bridge gap               = 2

overall recall = 0.9567
left recall    = 0.9318
right recall   = 0.9867
FPR            = 0.1341
```

It helps but does not pass the full gate. The original nearby-satellite search,
capped at 16 slices, does not improve recall.

## Stage 1b — remote satellite recovery

Stage 1b freezes the local R1 rule above and adds only one new degree of freedom:
a strongly supported farther satellite segment.

```text
R2b remote satellite:
  frozen B1 + at most one supported satellite

R3 hybrid:
  fixed Stage-1 local hysteresis + R2b
```

The 360-config validation sweep produced 40 feasible points: 20 R2b and 20 R3.
The recovery appears only once maximum satellite distance reaches 64 slices,
matching the Stage-1 topology audit.

Pre-MedSAM finalists:

| Candidate | Recall | Left | Right | FPR | Prompt success | Continuous |
|---|---:|---:|---:|---:|---:|---:|
| Frozen B1 | 0.9502 | 0.9275 | 0.9775 | 0.1193 | 0.8233 | 0.7787 |
| Fixed local hysteresis | 0.9567 | 0.9318 | 0.9867 | 0.1341 | 0.8274 | 0.7817 |
| R2b remote satellite | 0.9619 | 0.9488 | 0.9775 | **0.1193** | 0.8233 | 0.7819 |
| R3 hybrid | **0.9684** | **0.9531** | **0.9867** | 0.1341 | **0.8274** | **0.7849** |

`STAGE2_LOCK.json` freezes these exact four candidates before MedSAM evaluation.

## Stage 2 — negative-aware frozen MedSAM

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

R2b cleanly improves presence/detection recall while preserving negative
specificity. However, it does not improve case-volume Dice.

A paired slice diagnostic explains the mismatch. Relative to frozen B1, R2b
newly prompts 22 combined GT-positive slices and zero GT-negative slices. All 22
belong to `case_00067`. Those slices add only 650 TP pixels but 19,374 FP pixels;
their mean Dice is `0.0298` and median Dice is `0.0`. The `case_00067`
case-volume Dice decreases by about `0.01635`.

Therefore the remaining bottleneck is not simply presence. The network can
identify additional remote positive slices, but the frozen center/size geometry
on those recovered low-confidence slices is not reliable enough for MedSAM.

## Final decision

`FINAL_DECISION.json` freezes the V1a.4 conclusion:

- **no V1a.4 decoder replaces frozen V1a.3-B B1** under the predeclared Stage-2
  ranking rule;
- no V1a.4 secondary-test run is warranted;
- no post-Stage-2 decoder tuning is allowed;
- future work should target box/prompt geometry for recovered or low-confidence
  slices rather than widen the presence decoder again.

## Reproducibility

Compact results are under `results/`. Git intentionally excludes notebooks,
checkpoints, MedSAM weights, raw/processed KiTS23 data, NPZ files, and large
per-slice prediction/sweep tables.
