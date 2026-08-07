# V1a.5 design — geometry recovery for validation-restored kidney slices

## Motivation

V1a.4 showed that the long-range remote-satellite decoder can recover true
positive side-slices without increasing validation presence FPR. However, those
newly prompted slices do not improve the complete frozen-MedSAM system.

For the locked R2b finalist, relative to frozen B1:

```text
newly prompted combined GT-positive slices = 22
newly prompted combined GT-negative slices = 0
all 22 belong to case_00067
added TP pixels = 650
added FP pixels = 19,374
median recovered-slice Dice = 0
```

This isolates a new bottleneck: **presence recovery is correct, but the native
per-slice center/size prediction on the recovered low-confidence slices is not a
reliable MedSAM box**.

V1a.5 therefore changes geometry only. It does not widen presence recovery and
does not retrain any network.

## Frozen upstream

```text
V1a.4 merge       = 221f7f63a3a727b96c286edf09b9ad47e84bf555
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
expansion         = 1.20
target label      = kidney / 1
```

Presence is frozen to two references:

- G0 uses the V1a.3-B frozen B1 presence and geometry;
- G1/G2/G3 use the exact V1a.4 R2b presence mask selected before V1a.4 MedSAM.

R2b itself is frozen to:

```text
mode                     = remote_satellite
satellite max distance   = 64
satellite evidence ratio = 0.10
satellite min length     = 1
max satellites           = 1
```

## Stage-1 candidates

### G0 — frozen B1 reference

No change. This is the current validation system winner.

### G1 — R2b + native geometry

No geometry recovery. This reproduces the V1a.4 remote-satellite operating point.

### G2 — nearest reliable anchor propagation

For every side-slice that is present in R2b but absent from frozen B1:

1. keep the R2b presence decision;
2. find the nearest frozen-B1 positive slice for the same case and side;
3. copy that anchor's predicted `x1,y1,x2,y2` box onto the recovered slice.

No non-recovered box may change.

### G3 — local z-axis box trend

For every recovered side-slice:

1. keep the R2b presence decision;
2. select the five frozen-B1 positive anchors nearest in z for the same case/side;
3. convert each anchor box to `(cx, cy, width, height)`;
4. fit a linear trend versus slice index for each of the four parameters;
5. evaluate the trend at the recovered slice and clip the resulting box to the image.

For an internal recovered gap this behaves as interpolation. For a recovered
satellite outside the frozen B1 span it becomes bounded image-clipped linear
extrapolation from the nearest reliable anchor region.

The five-anchor count is fixed before Stage 1 and is not swept.

## Stage-1 safeguards

Stage 1 is validation-only and does **not** run MedSAM.

Hard invariants:

- 20 validation cases / 3,644 validation rows exactly;
- exact epoch-18 checkpoint SHA;
- G1/G2/G3 presence is byte-for-byte identical to frozen R2b;
- G2/G3 geometry on every non-recovered side-slice is unchanged from R2b;
- expansion remains 1.20;
- no test/secondary-test access.

Stage 1 evaluates both full-validation prompt metrics and the recovered-side
subset separately. A geometry method may advance only if it improves recovered
box IoU and recovered center error relative to G1 while preserving the frozen
presence decisions and non-recovered geometry.

## Stopping rule

- If neither G2 nor G3 improves recovered-slice prompt geometry, stop V1a.5 and
  do not run MedSAM.
- If one or both improve, freeze the small candidate set before any MedSAM result
  and run one negative-aware validation MedSAM comparison.
- No parameter change is allowed after a V1a.5 MedSAM comparison begins.
- The already-exposed 27-case split is not used for V1a.5 development.
