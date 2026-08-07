# V1a.3-A locked design

## Hypothesis

V1a.2-B repaired box geometry, but the complete automatic system is limited by
presence false positives. V1a.3-A tests whether the remaining problem is mainly
caused by training-example composition rather than by the network architecture
or presence loss.

## Single intended model change

Replace the V1a.2-B row sampler with a presence-focused sampler prioritizing:

1. false-positive or high-probability fully negative slices mined by V1a.2-B;
2. fully negative slices close to kidney onset and offset;
3. unilateral-positive slices, which provide a hard absent-side target;
4. a small retained set of far negatives.

## Frozen components

- V1a.2-B architecture;
- V1a.2-B loss and all loss weights;
- predicted-center size supervision;
- decoded-corner SmoothL1;
- 2.5D input and preprocessing;
- train/validation/test case split;
- seed 42;
- AdamW, cosine schedule, EMA, and augmentation;
- frozen MedSAM evaluation protocol.

## Development sequence

1. Generate train-split predictions from V1a.2-B epoch 14.
2. Audit presence errors and export a hard-negative pool.
3. Run the deterministic sampler gate.
4. Train V1a.3-A from scratch.
5. Export best-presence, best-geometry, and best-joint candidates.
6. Calibrate each candidate on validation only.
7. Run positive-slice and all-slice negative-aware MedSAM evaluation.
8. Compare with V1a.2-A locked and V1a.2-B epoch 14.

## Success criteria

A successful V1a.3-A candidate must:

- maintain overall presence recall at least 0.95;
- maintain left and right recall at least 0.90;
- improve negative specificity over V1a.2-B epoch 14;
- improve or match negative-aware case-volume Dice;
- preserve fixed-scale geometry within expected run-to-run variation.

## Stop condition

When targeted sampling lowers false positives but causes unacceptable recall or
geometry loss, do not add another box loss. The next controlled ablation should
change only the presence objective and become V1a.3-B.
