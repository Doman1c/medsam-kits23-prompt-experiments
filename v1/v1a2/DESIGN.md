# V1a.2-A Locked Design

## Motivation

Validation center-versus-size decomposition:

- center-only gain: 0.0149
- size-only gain: 0.0096
- center-size interaction: 0.0102

V1a.1 supervises size at the GT center but reads it at the predicted center during inference. V1a.2-A tests predicted-center size supervision as a single controlled change.

## Frozen settings

- 100 train / 20 validation / 27 untouched test cases
- 2.5D input, 256 input size, 64 heatmap size
- base channels 24, seed 42
- same sampler, augmentation, optimizer, EMA and checkpoint policy
- no test-set access during development

## Not included yet

No GIoU, decoded-box loss, small-area weighting, presence-loss change, threshold retuning or expansion retuning is included in V1a.2-A.
