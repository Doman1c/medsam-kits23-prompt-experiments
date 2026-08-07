# V1a.6 design — satellite-local geometry consensus

## Motivation

V1a.4 established that remote-satellite presence recovery can identify extra
GT-positive kidney slices without increasing validation presence FPR. V1a.5
then showed that distant-B1 geometry propagation is invalid for that remote
satellite: nearest reliable anchors are 49–73 slices away, and both copied and
linear-extrapolated boxes collapse recovered-slice overlap.

The remaining hypothesis is that the recovered satellite itself contains sparse
but useful native geometry signal. In V1a.5, 5/25 recovered slices had non-zero
native IoU and one slice reached IoU >= 0.5. V1a.6 tests whether robust local
consensus can stabilize that signal.

## Frozen upstream

```text
V1a.5 merge       = af2b92b96c0fecaecbfa21038b9d3aae3de27378
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
expansion         = 1.20
target label      = kidney / 1
```

Presence remains the exact V1a.4 R2b mask:

```text
satellite max distance   = 64
satellite evidence ratio = 0.10
satellite min length     = 1
max satellites           = 1
```

## Stage-1 candidates

### G1 — native R2b geometry

Reference only. No geometry modification.

### G4 — satellite-wide median

For each contiguous recovered satellite segment and side:

1. convert every native predicted box to `(cx, cy, width, height)`;
2. take the coordinate-wise median across the recovered segment;
3. use that one consensus geometry for all slices in the segment.

No GT information is used.

### G5 — rolling median

For each recovered slice, use a centered five-slice window within the recovered
satellite segment and take the coordinate-wise median `(cx, cy, width, height)`.
At the segment edges, use the available truncated window. Window size 5 is fixed
before evaluation and is not swept.

### G6 — robust local linear trend

For each recovered satellite segment:

1. represent native boxes as `(cx, cy, log(width), log(height))`;
2. compute the coordinate-wise median and MAD;
3. score each slice by its maximum robust normalized deviation;
4. retain inliers with score <= 3.5; if fewer than five survive, retain the five
   lowest-score slices;
5. fit a linear least-squares trend versus z independently for the four geometry
   parameters;
6. reconstruct and image-clip the box at each recovered slice.

The MAD threshold 3.5 and minimum five inliers are fixed before Stage 1 and are
not tuned after evaluation.

## Hard invariants

- validation split only: 20 cases / 3,644 rows;
- exact epoch-18 checkpoint SHA;
- G1/G4/G5/G6 presence must be identical to frozen R2b;
- only side-slices present in R2b and absent in frozen B1 may change geometry;
- all non-recovered geometry must remain byte-for-byte equal to R2b;
- expansion remains 1.20;
- no MedSAM before the geometry gate;
- no test or secondary-test access.

## Stage-1 advancement rule

Relative to G1 native geometry, a candidate advances only if both are true:

```text
recovered mean tight-box IoU increases
AND
recovered median normalized center error decreases
```

Recovered expanded coverage and prompt-success rate are supporting metrics, not
selection overrides.

## Stopping rule

- If none of G4/G5/G6 passes the geometry gate, stop V1a.6 before MedSAM.
- If one or more pass, freeze a minimal candidate set before any MedSAM result
  and run one negative-aware validation comparison.
- No post-gate sweep of window size, MAD threshold, inlier count, trend order,
  expansion, checkpoint, or presence decoder is allowed.
