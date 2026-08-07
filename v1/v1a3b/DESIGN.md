# V1a.3-B design — case-level presence span decoding

## Motivation

V1a.3-A changed only the training sampler. It improved negative-slice suppression
on validation while preserving V1a.2-B geometry, but the frozen test did not
become a new overall winner. V1a.3-B therefore does **not** retrain the image
network and does **not** change MedSAM. It asks whether explicit case-level
z-axis continuity can remove residual presence islands more reliably than
slice-wise thresholding plus local gap filling.

## Frozen inputs

V1a.3-B starts from the frozen V1a.3-A prompt network:

```text
checkpoint epoch = 18
weights          = EMA
input            = 2.5D
geometry         = V1a.2-B / V1a.3-A unchanged
MedSAM           = frozen
expansion        = 1.20 during initial decoder comparison
```

The V1a.3-A test split is not reused for V1a.3-B model or decoder selection.
Development returns to the 20-case validation split only.

## Controlled change

Only the bilateral presence decoding rule changes.

```text
B0: locked V1a.3-A smoothing + threshold + max-gap
B1: keep one dominant threshold-derived segment per side
B2: choose one optimal contiguous anatomical span per side
```

B1 can only delete extra islands. B2 may bridge a limited run of low-confidence
slices, but long low-confidence runs split the sequence and cannot be crossed.
Box center and size values are never altered by these decoders.

## Validation constraints

Candidate decoders must satisfy:

```text
overall positive-side recall >= 0.95
left positive-side recall    >= 0.90
right positive-side recall   >= 0.90
```

The initial sweep ranks feasible configurations by lower side-level false
positive rate. A small shortlist is then evaluated with the existing prompt
metrics. Frozen-MedSAM negative-aware evaluation is deliberately optional and
is enabled only after the shortlist is inspected.

## Stop rule

Do not access the test split during V1a.3-B development. If case-level decoding
cannot improve the validation negative-aware trade-off over V1a.3-A / V1a.2-A,
stop this branch rather than retuning the image network around the old test
result.
