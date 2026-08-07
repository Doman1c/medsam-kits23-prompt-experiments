# V1a.6 compact results

This directory contains compact validation summaries only. Checkpoints, NPZ data,
MedSAM weights, large per-slice prediction tables, notebooks, and test results are
excluded from Git.

## Stage 1

Satellite-local geometry consensus was compared on the 25 recovered validation
side-slices from `case_00067 / left`.

| Candidate | Recovered mean IoU | Median center error | Expanded coverage | Prompt success | Stage-1 gate |
|---|---:|---:|---:|---:|---|
| G1 native | 0.03285 | 5.2500 | 0.1957 | 0.000 | reference |
| G4 satellite median | 0.00284 | 3.6250 | 0.0587 | 0.000 | no |
| G5 rolling median | 0.02993 | 4.9286 | 0.1991 | 0.000 | no |
| G6 robust linear | **0.03606** | **4.1134** | **0.3363** | **0.040** | **yes** |

G6 was the only method satisfying the predeclared recovered-IoU and center-error
gate, so it advanced to the frozen negative-aware MedSAM comparison.

## Stage 2

The Stage-2 review bundle lists 16 compact files; all recorded sizes and SHA-256
hashes were verified. No test/secondary-test result, checkpoint, NPZ, or large
prediction table is included.

Predeclared ranking:

```text
case-volume Dice
-> all-slice mean Dice
-> negative-slice specificity
```

| Candidate | All-slice Dice | Positive Dice | Detection recall | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|---:|
| **G0 frozen B1** | 0.835611 | 0.810180 | 0.933812 | 0.848954 | **0.854752** |
| G1 R2b native | **0.835791** | **0.810703** | 0.951356 | 0.848954 | 0.853935 |
| G6 robust linear | 0.835782 | 0.810677 | **0.953748** | 0.848954 | 0.853540 |

G6 therefore does not become a system winner. On the 25 recovered slices, G6
versus G1 gives:

```text
mean Dice        0.02620 -> 0.02492
TP pixels             650 -> 608
FP pixels          19,374 -> 28,509
```

Paired recovered-slice counts are 6 improved, 6 worsened, and 13 unchanged. The
mean Dice delta is `-0.001279`, while FP pixels increase by 9,135.

## Final decision

Frozen V1a.3-B B1 remains the complete-system validation winner. V1a.6 stops
without a secondary-test run and without post-Stage-2 parameter tuning. This also
closes the image-only post-processing patch sequence: the next project phase
should synthesize/freeze the image-only result or move to a distinct model-level
or language-guided research question rather than opening another V1a.7-style
geometry tweak.
