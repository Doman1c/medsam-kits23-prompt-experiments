# V1a.4 design — recover recall without giving back V1a.3-B FP suppression

## Motivation

V1a.3-B established that a case-level single-span prior removes detached z-axis
false-positive islands. On validation it improved all-slice robustness and
negative specificity, and the locked secondary evaluation reproduced the same
false-positive suppression. However, the hard single-span prior also pruned true
positive kidney slices on the already-exposed secondary test.

V1a.4 asks a narrower controlled question:

> Can we recover slices near the frozen dominant kidney span while preserving a
> material fraction of V1a.3-B's false-positive suppression?

The V1a.3-B secondary test motivates this research question, but V1a.4 parameter
selection is validation-only. The secondary-test results must not be used to
choose V1a.4 thresholds, distances, gaps, or decoder family.

## Frozen upstream

No retraining is allowed in V1a.4 Stage 1.

- V1a.3-A epoch 18 EMA checkpoint;
- checkpoint SHA-256 `f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2`;
- frozen V1a.3-B B1 seed:
  - single span;
  - left threshold `0.0035`;
  - right threshold `0.0400`;
  - baseline max gap `4`;
  - boundary pad `1`;
- geometry unchanged;
- MedSAM unchanged;
- expansion `1.20` unchanged;
- target label `1` (kidney).

## Stage 0 — recovery audit

Before comparing new decoders, quantify which GT-positive slices are absent from
the frozen B1 seed on the 20-case validation split.

For every missed positive side-slice, record:

- before / after / inside / no-seed relation to the frozen dominant span;
- distance to the nearest frozen seed slice;
- side;
- smoothed presence probability.

The audit reports what fraction of misses lie within 1, 2, 4, 6, 8, 12, and 16
slices of the frozen span. This distinguishes boundary under-coverage from a
truly disconnected or completely missed side.

## Stage 1 decoder families

### R0 — frozen V1a.3-B B1

Reference only. No change.

### R1 — hysteresis boundary recovery

Use the frozen B1 span as the high-confidence seed. Expand each boundary only
outward, at a lower recovery threshold, up to a bounded maximum extension.
Optionally bridge a very short below-threshold run if stronger evidence resumes
immediately after it.

Properties:

- cannot create a disconnected new island;
- does not change center/size geometry;
- uses only model probabilities and the frozen predicted span;
- directly targets boundary false negatives.

Validation search dimensions:

```text
left recovery threshold  = 0.0005 .. 0.003
right recovery threshold = 0.0025 .. 0.03
max extension             = 2,4,6,8,12,16 slices
bridge gap                = 0,1,2 slices
```

### R2 — limited nearby satellite retention

Reconstruct the frozen-threshold B0 candidate segments, keep the same dominant
segment as V1a.3-B, and allow at most one additional segment when it is:

- close to the dominant segment;
- long enough;
- supported by a minimum fraction of the dominant segment's threshold-relative
  evidence.

The satellite is proposed only from the frozen threshold/gap mask. Distant
activation islands remain rejected.

Validation search dimensions:

```text
satellite max distance   = 2,4,6,8,12,16 slices
satellite evidence ratio = 0.05,0.10,0.20,0.35,0.50
satellite min length     = 1,2,3 slices
max satellites           = 1 (fixed)
```

A combined R1+R2 decoder is intentionally excluded from Stage 1. It is considered
only if the audit and separate R1/R2 results show complementary error recovery.
This keeps the first experiment interpretable and limits validation overfitting.

## Locked Stage-1 recovery gate

The gate is defined from validation baselines, not from the secondary test:

```text
overall side-slice recall >= 0.96
left recall               >= 0.94
right recall              >= 0.97
side-slice FPR             <= 0.15
```

Rationale: approximately recover the locked B0 validation recall while retaining
a material fraction of the V1a.3-B false-positive reduction.

For each family, Stage 1 materializes up to three diverse feasible operating
points:

1. lowest FPR;
2. highest recall;
3. lowest onset/offset boundary error.

Prompt geometry is then checked at expansion `1.20`. No MedSAM is run in Stage 1.

## Decision after Stage 1

- If neither R1 nor R2 passes the recovery gate, stop and inspect the recovery
  audit before adding complexity.
- If only one family passes, advance that family to negative-aware MedSAM.
- If both pass, compare a small predeclared shortlist through MedSAM.
- Only if R1 and R2 recover complementary errors should a hybrid decoder be
  introduced as a separate subsequent stage.

## Test policy

No V1a.4 test or secondary-test evaluation is allowed during development.
Because the existing 27-case split has already been exposed in earlier versions,
a later run on it can only be called a locked secondary evaluation. A
publication-quality final estimate requires a new untouched holdout from
previously unused KiTS23 cases.
