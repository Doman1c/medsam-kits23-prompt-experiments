# V1a.6 — Satellite-local geometry consensus

V1a.6 starts from the completed V1a.5 study. V1a.5 showed that geometry copied
or extrapolated from the distant frozen-B1 span is not useful for the remote
R2b satellite: the nearest B1 anchors are 49–73 slices away.

V1a.6 asks a narrower question:

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

## Stage-1 gate and result

A method may advance to negative-aware MedSAM only if, relative to G1, it:

1. improves recovered-slice mean tight-box IoU; and
2. lowers recovered-slice median normalized center error;
3. while passing all presence/non-recovered-geometry invariance checks.

The compact Stage-1 bundle passed all recorded size/SHA-256 checks and contained
no MedSAM output, test result, checkpoint, NPZ data, or large prediction table.
The recovered subset is exactly 25 GT-positive side-slices from
`case_00067 / left`, slices `19..43`.

| Candidate | Recovered mean IoU | Median center error | Expanded coverage | Prompt success | Advance? |
|---|---:|---:|---:|---:|---|
| G1 native | 0.03285 | 5.2500 | 0.1957 | 0.000 | reference |
| G4 satellite median | 0.00284 | **3.6250** | 0.0587 | 0.000 | no |
| G5 rolling median | 0.02993 | 4.9286 | 0.1991 | 0.000 | no |
| **G6 robust linear** | **0.03606** | 4.1134 | **0.3363** | **0.040** | **yes** |

G6 is the only predeclared method that improves both Stage-1 gate metrics. Its
improvement is modest, so `STAGE2_LOCK.json` froze exactly G0/G1/G6 before any
V1a.6 MedSAM output was opened.

## Stage 2 — negative-aware frozen MedSAM

Stage 2 compares exactly:

```text
G0 = frozen V1a.3-B B1 complete-system reference
G1 = frozen R2b + native geometry
G6 = frozen R2b + robust satellite-local linear geometry
```

Predeclared ranking:

```text
case-volume Dice
-> all-slice mean Dice
-> negative-slice specificity
```

The uploaded Stage-2 review bundle lists 16 compact files; every recorded size
and SHA-256 value was re-verified. The bundle contains no test/secondary-test
result, checkpoint, NPZ data, or large prediction table.

| Candidate | All-slice Dice | Positive Dice | Detection recall | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|---:|
| **G0 frozen B1** | 0.835611 | 0.810180 | 0.933812 | 0.848954 | **0.854752** |
| G1 R2b native | **0.835791** | **0.810703** | 0.951356 | 0.848954 | 0.853935 |
| G6 robust linear | 0.835782 | 0.810677 | **0.953748** | 0.848954 | 0.853540 |

G6 therefore does not become a complete-system winner. Relative to G1, it raises
positive-slice detection recall slightly but lowers case-volume Dice and does not
improve all-slice/positive Dice.

### Recovered-slice paired analysis

On the exact 25 recovered validation slices:

```text
G1 mean Dice = 0.026199
G6 mean Dice = 0.024921

G1 TP pixels =    650
G6 TP pixels =    608

G1 FP pixels = 19,374
G6 FP pixels = 28,509
```

Paired outcome:

```text
6 slices improve
6 slices worsen
13 slices are unchanged
mean Dice delta = -0.001279
TP delta        = -42
FP delta        = +9,135
FN delta        = +42
```

The Stage-1 geometry improvement therefore does not translate into better frozen
MedSAM segmentation. The robust trend can move boxes toward the target according
to prompt-level metrics while simultaneously producing larger or less useful
MedSAM regions.

## Final decision

`FINAL_DECISION.json` freezes the V1a.6 conclusion:

- frozen V1a.3-B B1 remains the complete-system validation winner;
- G6 does not replace G1 or G0;
- no V1a.6 secondary-test run is warranted;
- no post-Stage-2 tuning of MAD threshold, inlier count, trend form, expansion,
  or recovered-satellite rule is allowed;
- no V1a.7-style geometry-patching sequence is opened;
- the image-only V1a mainline is frozen for project synthesis.

The next project phase should either consolidate the image-only study into final
ablations/results/figures or open a genuinely distinct research question, such as
a learned cross-slice geometry model or language/semantic-guided spatial prompt
generation.

Compact results are stored under `results/`.
