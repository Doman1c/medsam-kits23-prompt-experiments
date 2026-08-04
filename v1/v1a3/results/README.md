# V1a.3-A results

V1a.3-A tests whether targeted example sampling can repair bilateral kidney
presence errors without changing the V1a.2-B model, optimized loss, data split,
or frozen MedSAM.

No notebook, model checkpoint, MedSAM weight, processed NPZ file, or large
per-slice prediction table is stored here. Compact machine-readable results are
provided in:

- `final_results.json`
- `validation_negative_aware_comparison.csv`
- `frozen_test_metrics.csv`
- `test_protocol_lock.json`

## Final training setup

The final sampler used 20,953 training rows and drew 13,527 samples per epoch.
Fully negative rows received 45% of the sampling probability.

| Sampling category | Available rows | Probability mass |
|---|---:|---:|
| Positive bilateral | 4,919 | 0.270827 |
| Positive unilateral | 2,521 | 0.279173 |
| Mined negative | 797 | 0.202500 |
| Boundary negative | 541 | 0.157500 |
| Near negative | 379 | 0.067500 |
| Far negative | 11,796 | 0.022500 |

Hard-negative mining retained 993 rows from 94 training cases. The final
settings were `boundary_radius=3`, `hard_negative_min_score=0.01`, a 1.5x
unilateral-positive boost, and seed 42.

Training stopped after 25 epochs. The prompt-network proxy checkpoints
`best_presence.pt`, `best_geometry.pt`, and `best_joint.pt` all pointed to epoch
13, but these proxy scores did not determine the final model.

## Validation-only selection

Checkpoints 10, 13, 18, and 25 were calibrated without using the test split.
Epoch 25 failed the locked recall constraints, epoch 10 was dominated by epoch
13, and epochs 13 and 18 entered frozen-MedSAM evaluation.

| Candidate | Positive-slice Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|---:|
| V1a.3-A epoch 13 | 0.8039 | 0.7462 | 0.8388 | 0.7159 |
| **V1a.3-A epoch 18** | **0.8067** | **0.7760** | **0.8479** | **0.7598** |

The predeclared ranking rule was case-volume Dice, then all-slice Dice, then
lower negative false-positive mask rate. Epoch 18 therefore became the locked
V1a.3-A candidate.

### Frozen validation configuration

```text
checkpoint epoch: 18
weights:          EMA
variant:          smooth
max gap:          2
left threshold:   0.005
right threshold:  0.005
expansion scale:  1.20
target label:     kidney = 1
```

Artifact hashes were locked before the first V1a.3-A test access:

```text
checkpoint:
f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2

calibration:
3d8c6baa4bcebe414fb2dc4aae023c1ce62366494b868ff7b06f65ae5d50f09f

final validation config:
20890ca3c55c56cd5d31cb64440c5ffb63d5f41ab6bc2eb3ecad7ed989d8b972
```

### Geometry preservation

With ground-truth presence and expansion fixed at 1.20, epoch 18 produced:

| Decomposition mode | Mean Dice |
|---|---:|
| M0 automatic center + automatic size | 0.8416 |
| M1 oracle center + automatic size | 0.8547 |
| M2 automatic center + oracle size | 0.8458 |
| M3 oracle center + oracle size | 0.8709 |

The joint oracle-box gap was `0.0293`. M0 was effectively unchanged from
V1a.2-B (`0.8416`), so targeted sampling did not collapse the learned box
geometry.

## Frozen test

The test protocol was locked before exporting predictions. It used 27 cases,
4,277 slices, epoch-18 EMA weights, smooth gap-2 presence, bilateral thresholds
of 0.005, and expansion 1.20. No test-based selection or post-test tuning was
performed.

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

The automatic-to-oracle slice-Dice gap was `0.1475`.

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

The manifest contains 1,473 positive test slices after the canonical
minimum-component-area filter. The all-slice evaluator counts 1,474
label-positive slices because `case_00293`, slice 513, contains one kidney-label
pixel that is intentionally excluded from the manifest track. This one-row
difference is documented rather than corrected after test access.

## Interpretation

V1a.3-A supports the controlled validation conclusion that targeted negative
sampling can improve negative-slice suppression while preserving V1a.2-B box
geometry. On the frozen test, however, automatic positive-slice Dice was
`0.7239`, below the V1a.1 frozen baseline (`0.7628`), and case-mean Dice was
`0.7627`, below V1a.1 (`0.7950`).

V1a.3-A is therefore recorded as a completed negative-sampling study, not as a
new overall system winner. The result is frozen as observed; no threshold,
checkpoint, smoothing, gap, expansion, sampler, or loss was changed after test
access.
