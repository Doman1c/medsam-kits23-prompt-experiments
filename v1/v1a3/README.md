# V1a.3-A — Presence repair by targeted sampling

V1a.3-A is a controlled follow-up to V1a.2-B. It tests whether bilateral kidney
presence errors are primarily caused by training-example composition.

The experiment keeps the V1a.2-B architecture, optimized loss, optimizer, EMA,
augmentation, model capacity, 2.5D input, case split, and frozen MedSAM
unchanged. The only intended modeling change is the training sampler.

## Controlled change

Unchanged from V1a.2-B:

- bilateral presence heads;
- heatmap center prediction;
- predicted-center size supervision;
- normalized decoded-corner SmoothL1;
- 2.5D input;
- optimizer, EMA, augmentation, model capacity, and case split.

Changed in V1a.3-A:

- mined hard-negative oversampling;
- kidney-span boundary-negative oversampling;
- near-negative retention;
- limited far-negative sampling;
- unilateral-positive oversampling to strengthen absent-side supervision.

## Final sampler

Each training row belongs to exactly one category:

```text
positive_bilateral
positive_unilateral
negative_mined
negative_boundary
negative_near
negative_far
```

The final executed configuration was:

```text
fully negative sampling mass = 45%

within negative mass:
  mined hard negatives = 45%
  boundary negatives   = 35%
  near negatives       = 15%
  far negatives        = 5%

boundary radius              = 3 slices
hard-negative minimum score  = 0.01
hard-negative top-k per case = 32
unilateral-positive boost    = 1.50
seed                          = 42
```

The final training set contained 20,953 rows. Hard-negative mining retained 993
rows from 94 cases. The realized category masses matched the requested
composition.

## Training

Training ran from scratch for 25 epochs and stopped after 12 epochs without a
new prompt-network proxy winner. `best_presence.pt`, `best_geometry.pt`, and
`best_joint.pt` all pointed to epoch 13.

Those files were only training-time proxies. Final selection remained
validation-only and used calibrated frozen-MedSAM results.

## Validation selection

Checkpoints 10, 13, 18, and 25 underwent presence calibration. Epoch 25 failed
the locked recall constraints, epoch 10 was dominated by epoch 13, and epochs 13
and 18 entered positive-slice and negative-aware frozen-MedSAM validation.

| Candidate | Positive-slice Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|---:|
| Epoch 13 | 0.8039 | 0.7462 | 0.8388 | 0.7159 |
| **Epoch 18** | **0.8067** | **0.7760** | **0.8479** | **0.7598** |

The predeclared ranking rule was:

1. higher case-volume Dice;
2. higher all-slice mean Dice;
3. lower negative false-positive mask rate.

Epoch 18 became the unique frozen validation winner.

### Locked configuration

```text
checkpoint epoch: 18
weights:          EMA
variant:          smooth
max_gap:          2
left threshold:   0.005
right threshold:  0.005
expansion scale:  1.20
target label:     kidney = 1
```

## Frozen test

The test protocol was locked before exporting any V1a.3-A test predictions.
The test contained 27 cases and 4,277 slices. No checkpoint, threshold,
smoothing, max-gap, expansion, sampler, or loss was changed after test access.

### Prompt metrics

| Metric | Result |
|---|---:|
| Presence accuracy | 0.8325 |
| Positive-side recall | 0.9398 |
| Left recall | 0.9479 |
| Right recall | 0.9291 |
| Side false-positive rate | 0.2084 |
| Tight-box IoU | 0.7089 |
| Prompt success rate | 0.7596 |

### Positive-slice frozen MedSAM

| Mode | Slice Dice | Case-mean Dice |
|---|---:|---:|
| Automatic smooth prompts | 0.7239 | 0.7627 |
| Oracle scaled boxes | 0.8713 | 0.8773 |

Automatic-to-oracle slice-Dice gap: `0.1475`.

### Negative-aware all-slice frozen MedSAM

| Metric | Result |
|---|---:|
| All-slice mean Dice | 0.7475 |
| All-slice case-mean Dice | 0.7846 |
| Case-volume Dice | 0.7996 |
| Positive-slice Dice | 0.7234 |
| Positive-slice detection recall | 0.9715 |
| Negative specificity | 0.7603 |
| Negative false-positive mask rate | 0.2397 |

The canonical manifest contains 1,473 positive test slices. The all-slice
evaluator counts 1,474 label-positive slices because `case_00293`, slice 513,
contains one target-label pixel below the manifest's minimum-component-area
filter. This discrepancy is documented and was not used to alter the frozen
evaluation.

## Interpretation

V1a.3-A supports the validation conclusion that targeted negative sampling
improves negative-slice suppression while preserving V1a.2-B geometry. The
epoch-18 fixed-presence geometry score was effectively unchanged from V1a.2-B:

```text
M0 automatic center + automatic size = 0.8416
joint oracle-box gap                  = 0.0293
```

The frozen test did not establish V1a.3-A as a new overall winner. Its automatic
positive-slice Dice (`0.7239`) and case-mean Dice (`0.7627`) were below the
V1a.1 frozen baseline (`0.7628` and `0.7950`).

The correct conclusion is therefore:

> Targeted sampling repaired part of the presence/negative-slice failure mode
> on validation without damaging box geometry, but the improvement did not
> translate into a stronger frozen-test positive-slice system.

## Results committed to Git

Compact results are under [`results/`](results/):

```text
results/
├── README.md
├── final_results.json
├── validation_negative_aware_comparison.csv
├── frozen_test_metrics.csv
└── test_protocol_lock.json
```

The repository intentionally excludes notebooks, model weights, processed NPZ
slices, and large per-slice prediction tables.
