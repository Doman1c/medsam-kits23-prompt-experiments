# MedSAM Automatic Prompt Generation on KiTS23

This repository studies a lightweight, modular, and interpretable automatic
spatial-prompt generator for **frozen MedSAM** on KiTS23 kidney CT.

```text
2.5D CT
  -> bilateral kidney presence
  -> bilateral box geometry
  -> case-level prompt decoding
  -> frozen MedSAM
  -> kidney segmentation
```

## Project status

The **image-only V1a mainline is frozen after V1a.6**. The project no longer
continues with V1a.7-style post-hoc geometry or threshold patches.

The final frozen robust operating point is **V1a.3-B B1**, using the V1a.3-A
epoch-18 EMA prompt network plus a case-level single-span presence decoder.
V1a.4–V1a.6 remain important controlled failure-analysis experiments, but none
replaces the frozen B1 system.

For the full research narrative, see:

- [V1a mainline synthesis](v1/V1A_MAINLINE_SYNTHESIS.md)
- [V1a final lock](v1/V1A_FINAL_LOCK.json)
- [Comparable V1a ablation table](v1/results/v1a_mainline_ablation.csv)

## Final frozen V1a operating point

```text
prompt network:
  source            = V1a.3-A
  checkpoint         = epoch 18 EMA
  checkpoint SHA256  = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2

case-level decoder:
  source            = V1a.3-B B1
  mode              = single_span
  left threshold    = 0.0035
  right threshold   = 0.0400
  baseline max gap  = 4
  boundary pad      = 1

MedSAM:
  frozen            = true
  expansion scale   = 1.20
  target label      = 1
```

Validation metrics:

| Metric | Frozen B1 |
|---|---:|
| Positive-slice Dice | 0.810180 |
| All-slice Dice | 0.835611 |
| Case-volume Dice | 0.854752 |
| Negative specificity | 0.848954 |
| Positive-slice detection recall | 0.933812 |

A useful nuance is that V1a.2-A retains the highest historical validation
case-volume mean (`0.855560`). Frozen B1 is kept as the final robust operating
point because it raises all-slice Dice from `0.766569` to `0.835611` and negative
specificity from `0.744351` to `0.848954`, with only a small case-volume change.

## Experiment map

| Version | Main question | Status |
|---|---|---|
| [V0](v0/) | Which frozen-MedSAM prompt formulation is strongest? | Complete |
| [V0.5](v0.5/) | How robust is MedSAM to imperfect boxes? | Complete |
| [V1a.1](v1/v1a1/) | Can a 2.5D dual-kidney model generate automatic boxes? | Frozen evaluation complete |
| [V1a.2-A](v1/v1a2/) | Can predicted-center size supervision repair train/inference mismatch? | Validation complete |
| [V1a.2-B](v1/v1a2b/) | Does decoded-corner supervision improve box geometry? | Validation complete |
| [V1a.3-A](v1/v1a3/) | Can targeted negative sampling repair presence? | Frozen model-specific evaluation complete |
| [V1a.3-B](v1/v1a3b/) | Can case-level z continuity suppress detached false positives? | Final frozen image-only operating point |
| [V1a.4](v1/v1a4/) | Can remote presence recovery regain over-pruned positives? | Validation complete — no new winner |
| [V1a.5](v1/v1a5/) | Can geometry be propagated from the distant dominant span? | Validation complete — stopped before MedSAM |
| [V1a.6](v1/v1a6/) | Can the recovered satellite stabilize its own geometry? | Validation complete — no new winner |

## Main findings

### V0 / V0.5: box prompting is the correct foundation

Representative V0 random-100 MedSAM results:

| Prompt mode | Mean Dice |
|---|---:|
| box_mask | 0.9460 |
| box | 0.9414 |
| all | 0.9272 |
| box_pos | 0.8475 |
| box_pos_neg | 0.6240 |

V0.5 shows that box expansion around `1.10–1.25` is generally safe, small
center/corner errors are often tolerable, large errors cause sharp degradation,
and very small edge slices remain difficult.

### V1a.1–V1a.2: geometry is important, but positive-only metrics are unsafe

V1a.1's frozen positive-slice Dice is `0.7628` versus oracle-scaled `0.8713`.
Oracle-gap decomposition attributes most of the early gap to automatic box
localization/size.

V1a.2-A and V1a.2-B improve geometry, but negative-aware evaluation reveals that
a positive-slice prompt winner can be catastrophically poor on fully negative
slices. This motivates presence-aware model selection.

### V1a.3: case-level z continuity provides the largest robustness gain

Targeted negative sampling helps but does not fully solve presence generalization.
The V1a.3-B single-span decoder then suppresses detached false-positive islands
without retraining, producing the final robust operating point.

On the already-exposed 27-case split, the locked B1 secondary evaluation raises
all-slice Dice and negative specificity but reduces positive-slice recall. This
trade-off motivates the controlled V1a.4–V1a.6 recovery studies.

### V1a.4–V1a.6: recovering presence is not enough without usable geometry

V1a.4 successfully recovers true-positive remote slices without increasing
presence FPR, but their native boxes do not help MedSAM. V1a.5 shows that geometry
cannot be borrowed from reliable boxes 49–73 slices away. V1a.6 improves some
prompt-level geometry metrics using satellite-local robust trends, but downstream
MedSAM still does not improve.

This closes the image-only patching sequence and establishes an important final
observation: **better prompt-level box metrics do not necessarily imply better
MedSAM segmentation**.

## Evaluation protocol and caveat

The learned pipeline uses 147 selected KiTS23 cases with a fixed 100 train / 20
validation / 27 test split. Development is validation-only; MedSAM remains frozen.
Later studies include fully negative slices and use negative-aware system metrics.

The 27-case split was exposed earlier in the project. V1a.3-B therefore uses a
locked **secondary retrospective evaluation** on the same split, not a new
project-level pristine holdout. A publication-grade final generalization claim
should use a previously unused KiTS23 holdout or an external dataset after the
full method is frozen.

## Repository structure

```text
.
├── v0/
├── v0.5/
└── v1/
    ├── V1A_MAINLINE_SYNTHESIS.md
    ├── V1A_FINAL_LOCK.json
    ├── results/v1a_mainline_ablation.csv
    ├── v1a1/
    ├── v1a2/
    ├── v1a2b/
    ├── v1a3/
    ├── v1a3b/
    ├── v1a4/
    ├── v1a5/
    └── v1a6/
```

Git contains source code, locked configurations, compact comparison tables, and
summary documents. Model checkpoints, MedSAM weights, processed NPZ slices, raw
KiTS23 volumes, and large per-slice prediction tables are intentionally excluded.

## Next research phase

The next phase should be a genuinely distinct question rather than another V1a
patch. The preferred direction is **V1b language/semantic-guided spatial
prompting**, with learned cross-slice geometry as a separate alternative research
branch if pursued later.
