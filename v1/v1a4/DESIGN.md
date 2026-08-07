# V1a.4 design — recall recovery after hard single-span pruning

## Frozen upstream

V1a.4 starts from the merged V1a.3-B experiment and does not change the prompt
network, box geometry, MedSAM, target label, or expansion.

```text
V1a.3-B merge     = 36c260096f733fc54e4db0819cc04527d021eff6
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2

frozen B1 seed:
left threshold      = 0.0035
right threshold     = 0.0400
baseline max gap    = 4
boundary pad        = 1
expansion scale     = 1.20
```

The V1a.3-B secondary test is already exposed. It may motivate the V1a.4
question but cannot select V1a.4 parameters.

## Primary question

Can a small amount of case-level presence recovery regain slices pruned by the
hard single-span prior while preserving most of its false-positive suppression?

## Locked validation gate

All V1a.4 decoder selection uses only the 20-case / 3,644-row validation split.
The gate is fixed across development stages:

```text
overall recall >= 0.96
left recall    >= 0.94
right recall   >= 0.97
FPR             <= 0.15
```

MedSAM is not run until at least one decoder passes this presence-level gate.

## Stage 1

Stage 1 compares two local mechanisms against frozen B1:

- R1: bounded lower-threshold hysteresis extension from the dominant span;
- R2: retain at most one nearby threshold-derived satellite segment.

### Stage-1 result

Frozen B1 has recall `0.9502` and FPR `0.1193`. The audit found 107 FN slices:
83 before the seed, 24 after; 85 left and 22 right. Only 47/107 are within 16
slices of the seed.

A single case-side (`case_00067`, left) contributes 76 FN slices extending up to
76 slices before the selected dominant span. This is a split-span / dominant
segment selection failure rather than only a local boundary miss.

The best local hysteresis point under FPR 0.15 is frozen for subsequent use:

```text
left recovery threshold  = 0.0005
right recovery threshold = 0.0025
max extension            = 6
bridge gap               = 2

overall recall = 0.9567
left recall    = 0.9318
right recall   = 0.9867
FPR            = 0.1341
```

It helps but does not meet the gate. The Stage-1 satellite search was limited to
16 slices and produced no recall improvement. Stage 1 therefore rejects a
local-only repair as sufficient.

## Stage 1b

Stage 1b changes only the satellite support range. It does not re-sweep the local
hysteresis parameters.

Two families are predeclared:

### R2b remote satellite

Keep frozen B1 and allow at most one additional threshold-derived segment when:

- it is within a bounded but larger z-distance (`20,24,32,48,64,96` slices);
- its threshold-relative evidence is a sufficient fraction of the dominant span;
- it meets a minimum segment length.

The gap between the dominant span and satellite is not filled.

### R3 hybrid

Union the fixed Stage-1 R1 output with R2b. This is justified because Stage 1
showed that local hysteresis recovers one error class while the audit exposed a
separate long-range split-span class.

Satellite sweep dimensions:

```text
distance        = 20,24,32,48,64,96
evidence ratio  = 0.05,0.10,0.20,0.35,0.50,0.75
minimum length  = 1,2,3,5,8
max satellites  = 1
```

The validation gate remains unchanged. No MedSAM and no test access occur in
Stage 1b.

## Stopping rule

- If neither R2b nor R3 passes the locked gate, stop V1a.4 presence recovery;
  do not keep widening the decoder around validation failures.
- If one or more configurations pass, retain a small diverse validation
  shortlist and then run one negative-aware MedSAM comparison.
- Any later locked evaluation must use one configuration selected entirely on
  validation and must not trigger post-evaluation parameter changes.
