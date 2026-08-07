# V1a.5 — Geometry recovery for restored kidney slices

V1a.5 starts from the completed V1a.4 validation study. The key V1a.4 finding
was that remote-satellite presence recovery can identify additional true-positive
kidney slices without increasing validation presence FPR, but the native box on
those low-confidence slices is too poor for frozen MedSAM.

V1a.5 therefore asks:

> When case-level context says a slice should be prompted, can its spatial box be
> recovered from reliable z-axis geometry instead of trusting that slice's native
> center/size prediction?

No retraining is performed.

## Frozen upstream

```text
V1a.4 merge       = 221f7f63a3a727b96c286edf09b9ad47e84bf555
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
expansion         = 1.20
```

The V1a.4 R2b presence decoder is frozen:

```text
satellite max distance   = 64
satellite evidence ratio = 0.10
satellite min length     = 1
max satellites           = 1
```

## Stage-1 comparison

```text
G0 = frozen V1a.3-B B1

G1 = frozen V1a.4 R2b
     + native per-slice geometry

G2 = frozen V1a.4 R2b
     + nearest reliable B1-anchor box propagation

G3 = frozen V1a.4 R2b
     + five-anchor z-axis linear box trend
```

Only side-slices newly recovered by R2b are allowed to change geometry in G2/G3.
Presence and every non-recovered box remain frozen.

The predeclared advancement rule is strict: a geometry method may advance to
negative-aware MedSAM only if it improves both recovered-slice tight-box IoU and
recovered-slice median center error relative to G1.

## Stage-1 result

The review bundle contained 20 listed result files; all recorded sizes and
SHA-256 values were verified. No MedSAM result, test result, checkpoint, NPZ, or
large prediction table was included.

The R2b-recovered subset contains 25 positive side-slices, all from
`case_00067 / left`, slices `19..43`.

| Candidate | Recovered IoU mean | Recovered center-error median | Recovered coverage | Prompt success | Advance? |
|---|---:|---:|---:|---:|---|
| G1 R2b native | 0.03285 | 5.2500 | 0.1957 | 0.000 | reference |
| G2 nearest anchor | 0.00000 | 7.3611 | 0.0000 | 0.000 | no |
| G3 linear trend | 0.00000 | **4.6887** | 0.0000 | 0.000 | no |

Global validation presence is unchanged across G1/G2/G3:

```text
positive recall = 0.96186
FPR             = 0.11931
```

The nearest frozen-B1 anchor is extremely far from the remote recovered segment:

```text
minimum distance = 49 slices
median distance  = 61 slices
maximum distance = 73 slices
```

This explains the failure mode. G2 copies a box from a spatially different distant
anchor region and loses all overlap. G3 improves the normalized center-error
median slightly through extrapolation, but every recovered slice still has zero
tight-box IoU and zero expanded coverage.

The native G1 geometry is poor but not uniformly useless: 5/25 recovered slices
have non-zero IoU and one slice reaches IoU >= 0.5. G2/G3 remove even that limited
overlap.

## Final decision

V1a.5 stops at Stage 1 under the predeclared rule:

- neither G2 nor G3 improves recovered-slice IoU versus G1;
- no V1a.5 negative-aware MedSAM run is performed;
- no V1a.5 secondary-test run is performed;
- no post-Stage-1 tuning of anchor count, propagation distance, or trend model is
  allowed;
- frozen V1a.3-B B1 remains the current complete-system validation winner.

The conclusion is frozen in `FINAL_DECISION.json`. Compact Stage-1 results are
stored under `results/`.

## Interpretation

V1a.4 established that the remote presence signal is useful. V1a.5 shows that
**geometry from the distant frozen-B1 span is not a valid source for recovering a
remote satellite box**. The next geometry experiment, if pursued, should use
information local to the recovered satellite itself (for example robust
within-satellite geometry consensus) or a learned/cross-slice geometry mechanism,
rather than continuing to extrapolate from anchors 49-73 slices away.

## Development rule

The V1a.3-B/V1a.4 27-case secondary split is already exposed and was not used for
V1a.5 selection. V1a.5 is validation-only. See `DESIGN.md`, `BASELINE_LOCK.json`,
and `FINAL_DECISION.json` for the frozen experiment definition and stopping rule.
