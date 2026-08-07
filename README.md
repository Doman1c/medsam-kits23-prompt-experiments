# MedSAM Automatic Prompt Generation on KiTS23

This repository studies a lightweight, modular, and interpretable automatic
spatial-prompt generator for frozen MedSAM on the KiTS23 kidney CT dataset.
The prompt model predicts bilateral kidney presence and bounding boxes from
2.5D CT input; MedSAM itself remains frozen.

```text
2.5D CT slices
    -> bilateral presence / center / size prediction
    -> calibrated spatial prompts
    -> frozen MedSAM
    -> kidney segmentation
```

## Research objective

The project asks two related questions:

1. Can a compact prompt generator replace manually supplied MedSAM boxes?
2. Which errors dominate the automatic system: presence decisions, box center,
   box size, or center-size interaction?

Development follows controlled validation-only ablations. Test/secondary-test
results are not used to select checkpoints, thresholds, smoothing, max-gap,
expansion, samplers, loss variants, or later decoder parameters.

## Experiment versions

| Version | Main question | Status |
|---|---|---|
| [V0](v0/) | Which MedSAM prompt combinations work best? | Complete |
| [V0.5](v0.5/) | How robust is MedSAM to prompt degradation? | Complete |
| [V1a.1](v1/v1a1/) | Can a 2.5D dual-kidney network generate automatic boxes? | Frozen test complete |
| [V1a.2-A](v1/v1a2/) | Does predicted-center size supervision repair train-inference mismatch? | Validation complete |
| [V1a.2-B](v1/v1a2b/) | Does decoded-corner supervision reduce joint center-size error? | Validation complete |
| [V1a.2 follow-up](v1/v1a2b/followup_validation/) | Which model survives fully negative slices and checkpoint-Pareto analysis? | Complete |
| [V1a.3-A](v1/v1a3/) | Can targeted negative sampling repair presence without changing geometry or loss? | Frozen test complete |
| [V1a.3-B](v1/v1a3b/) | Can case-level single-span decoding suppress detached presence islands? | Secondary evaluation complete |
| [V1a.4](v1/v1a4/) | Can controlled recovery regain recall lost by hard single-span pruning? | **Validation complete — no new winner** |

## V1a.2 findings

### Negative-aware all-slice validation

The follow-up evaluation contains 3,644 validation slices, including 2,390
fully negative slices.

| Operating point | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|
| V1a.1 locked | 0.3097 | 0.7112 | 0.0464 |
| V1a.2-A locked | 0.7666 | **0.8556** | 0.7444 |
| V1a.2-B epoch 4 prompt-best | 0.3060 | 0.7534 | 0.0335 |
| V1a.2-B epoch 14 calibrated | **0.7725** | 0.8508 | **0.7598** |

V1a.2-B is the strongest geometry design, while V1a.2-A remains slightly
better in validation case-volume Dice.

### Fixed-scale geometry decomposition

All models were compared with ground-truth presence and expansion fixed at
`1.20`.

| Model | M0 auto box | Joint box gap | Center-size interaction |
|---|---:|---:|---:|
| V1a.1 | 0.8361 | 0.0347 | 0.0102 |
| V1a.2-A | 0.8406 | 0.0302 | 0.0128 |
| V1a.2-B | **0.8416** | **0.0292** | **0.0069** |

## Completed V1a.3-A study

V1a.3-A retained the V1a.2-B architecture and optimized loss and changed only
the training sampler.

Validation selected epoch 18 with EMA weights, smooth presence, `max_gap=2`,
left/right thresholds `0.005/0.005`, and expansion `1.20`.

| Metric | V1a.3-A validation |
|---|---:|
| Positive-slice Dice | 0.8067 |
| All-slice Dice | 0.7760 |
| Case-volume Dice | 0.8479 |
| Negative specificity | 0.7598 |

The fixed-presence M0 geometry score was `0.8416`, showing that targeted
sampling did not collapse V1a.2-B box geometry.

The 27-case frozen test gave positive-slice Dice `0.7239`, all-slice Dice
`0.7475`, case-volume Dice `0.7996`, and negative specificity `0.7603`.

## Completed V1a.3-B study

V1a.3-B froze the V1a.3-A epoch-18 EMA network and changed only the case-level
presence decoder. Validation selected a simple single-span decoder:

```text
left threshold      = 0.0035
right threshold     = 0.0400
baseline max gap    = 4
boundary pad        = 1
expansion scale     = 1.20
```

Validation improved from B0 to B1 as follows:

| Metric | B0 | B1 frozen |
|---|---:|---:|
| All-slice Dice | 0.7760 | 0.8356 |
| Positive-slice Dice | 0.8067 | 0.8102 |
| Negative specificity | 0.7598 | 0.8490 |
| Case-volume Dice | 0.8479 | 0.8548 |

On the already-exposed 27-case split, the locked B1 secondary evaluation kept
the false-positive benefit but over-pruned positive slices:

| Metric | B0 | B1 frozen |
|---|---:|---:|
| All-slice Dice | 0.7475 | 0.8009 |
| Positive-slice Dice | 0.7234 | 0.6827 |
| Positive-slice detection recall | 0.9715 | 0.8636 |
| Negative specificity | 0.7603 | 0.8630 |
| Case-volume Dice | 0.7996 | 0.8016 |

V1a.3-B is therefore retained as a conservative operating point with strong
negative-slice suppression rather than an unqualified replacement for B0.

## Completed V1a.4 validation study

V1a.4 kept the V1a.3-A epoch-18 EMA network and the V1a.3-B frozen single-span
seed. It tested whether missed positive slices could be recovered without giving
back the false-positive suppression.

Stage 1 showed that the failure was not purely local: 107 frozen-B1 false-negative
side-slices were found, and one left case-side (`case_00067`) contributed 76 of
them, extending up to 76 slices before the selected dominant span.

Stage 1b therefore tested a strongly supported remote satellite segment. It
successfully crossed the locked presence gate. The cleanest R2b point raised
presence recall from `0.9502` to `0.9619` while keeping FPR fixed at `0.1193`.

Negative-aware frozen-MedSAM validation then compared four candidates under the
predeclared ranking rule: case-volume Dice, then all-slice Dice, then lower
negative FP-mask rate.

| Candidate | All-slice Dice | Positive Dice | Detection recall | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|---:|
| **Frozen V1a.3-B B1** | 0.83561 | 0.81018 | 0.93381 | **0.84895** | **0.85475** |
| Fixed local hysteresis | 0.82979 | 0.81400 | 0.93541 | 0.83808 | 0.85447 |
| R2b remote satellite | **0.83579** | 0.81070 | **0.95136** | **0.84895** | 0.85393 |
| R3 hybrid | 0.82997 | **0.81453** | **0.95295** | 0.83808 | 0.85365 |

R2b demonstrates that presence recovery itself works, but the recovered remote
slices do not have sufficiently reliable frozen box geometry. Relative to frozen
B1, R2b newly prompts 22 GT-positive combined slices (all from `case_00067`) and
no GT-negative slices, yet those slices add only 650 TP pixels and 19,374 FP
pixels; their median slice Dice is `0.0`.

Therefore V1a.4 does **not** replace frozen V1a.3-B B1. No V1a.4 secondary-test
run is performed, and no post-Stage-2 decoder tuning is allowed. The next
research bottleneck is geometry/prompt quality on recovered or low-confidence
slices, not further widening of case-level presence post-processing.

See [V1a.4 protocol and compact results](v1/v1a4/).

## Repository structure

```text
.
├── v0/
├── v0.5/
└── v1/
    ├── v1a1/
    ├── v1a2/
    ├── v1a2b/
    ├── v1a3/
    ├── v1a3b/
    └── v1a4/
```

## Reproducibility

Git contains source code, locked configurations, compact comparison tables, and
summary documents. The repository intentionally excludes notebooks used only for
Colab recovery, model checkpoints, MedSAM weights, raw KiTS23 volumes, processed
NPZ slices, and large per-slice prediction/sweep tables.
