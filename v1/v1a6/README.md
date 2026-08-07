# V1a.6 — Satellite-local geometry consensus

V1a.6 starts from the completed V1a.5 study. V1a.5 showed that geometry copied
or extrapolated from the distant frozen-B1 span is not useful for the remote
R2b satellite: the nearest B1 anchors are 49–73 slices away.

V1a.6 therefore asks a narrower question:

> Can the recovered satellite reconstruct a stable MedSAM box from its own
> native per-slice geometry predictions, without borrowing geometry from the
> distant dominant span?

No retraining is performed.

## Frozen upstream

```text
V1a.5 merge       = af2b92b96c0fecaecbfa21038b9d3aae3de27378
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
expansion         = 1.20

R2b presence:
satellite max distance   = 64
satellite evidence ratio = 0.10
satellite min length     = 1
max satellites           = 1
```

## Stage-1 candidates

```text
G1 = R2b + native per-slice geometry          [reference]
G4 = satellite-wide median geometry
G5 = within-satellite rolling median, window 5
G6 = satellite-local robust linear geometry trend
```

All G4/G5/G6 methods use only the native predicted boxes inside each recovered
satellite segment. Ground truth is used only for validation evaluation, never to
construct a candidate box.

Only R2b-recovered side-slices may change geometry. R2b presence and every
non-recovered box remain frozen.

## Stage-1 gate

A method may advance to negative-aware MedSAM only if, relative to G1, it:

1. improves recovered-slice mean tight-box IoU; and
2. lowers recovered-slice median normalized center error;
3. while passing all presence/non-recovered-geometry invariance checks.

No MedSAM and no test/secondary-test split are used before this gate.

## Stage-1 result

The uploaded compact bundle passed all recorded size/SHA-256 checks. It contains
no MedSAM output, test result, checkpoint, NPZ data, or large prediction table.

The recovered subset is again exactly 25 GT-positive side-slices from
`case_00067 / left`, slices `19..43`.

| Candidate | Recovered mean IoU | Median center error | Expanded coverage | Prompt success | Advance? |
|---|---:|---:|---:|---:|---|
| G1 native | 0.03285 | 5.2500 | 0.1957 | 0.000 | reference |
| G4 satellite median | 0.00284 | **3.6250** | 0.0587 | 0.000 | no |
| G5 rolling median | 0.02993 | 4.9286 | 0.1991 | 0.000 | no |
| **G6 robust linear** | **0.03606** | 4.1134 | **0.3363** | **0.040** | **yes** |

G4 substantially reduces center error but collapses overlap. G5 is almost
neutral. G6 is the only predeclared method that improves both gate metrics; it
also raises recovered expanded coverage and creates one recovered prompt-success
case. The effect is modest, so Stage 2 is needed to determine whether the box
change improves the complete frozen-MedSAM system.

Compact Stage-1 results are stored under `results/`.

## Frozen Stage 2

`STAGE2_LOCK.json` was written before opening any V1a.6 MedSAM result. Stage 2
compares exactly three validation candidates:

```text
G0 = frozen V1a.3-B B1 complete-system reference
G1 = frozen R2b + native geometry
G6 = frozen R2b + robust satellite-local linear geometry
```

Ranking remains:

```text
case-volume Dice
-> all-slice mean Dice
-> negative-slice specificity
```

No test or secondary-test split is accessed, and no parameter change is allowed
after the Stage-2 MedSAM comparison begins.

## Run Stage 2

If the current Colab runtime still contains the V1a.6 Stage-1 outputs:

```bash
cd /content/medsam-kits23-prompt-experiments

git fetch origin
git checkout -B \
  v1a6-satellite-local-geometry \
  origin/v1a6-satellite-local-geometry

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a6/scripts/run_v1a6_stage2_medsam.sh
```

Outputs:

```text
/content/v1a1_work/v1a6_stage2_medsam/
/content/drive/MyDrive/medsam-kits23/v1a6_results/stage2_medsam/
```

Package the compact review bundle:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a6/scripts/package_v1a6_stage2.py \
  --root /content/v1a1_work/v1a6_stage2_medsam \
  --output /content/v1a6_stage2_review.zip
```

Upload `v1a6_stage2_review.zip` before any further system decision.

## Development rule

The already-exposed 27-case split is not used for V1a.6 selection. All V1a.6
selection is validation-only. The satellite median, rolling-window size,
MAD threshold, inlier count, and linear trend form are fixed and are not swept
after seeing Stage-1 or Stage-2 results.
