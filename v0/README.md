# V0 — MedSAM Prompt Baseline

V0 evaluates frozen MedSAM using manually derived and oracle spatial prompts on KiTS23 kidney CT slices.

## Main scripts

- `prepare_kits23_2d.py`
- `run_v0_experiment.py`
- `environment_check.py`

## Prompt modes

- `box`
- `box_pos`
- `box_pos_neg`
- `box_mask`
- `all`

A representative random-100 experiment produced:

| Prompt mode | Mean Dice |
|---|---:|
| box_mask | 0.9460 |
| box | 0.9414 |
| all | 0.9272 |
| box_pos | 0.8475 |
| box_pos_neg | 0.6240 |

These experiments established bounding boxes as the most reliable prompt type for the learned automatic prompt-generation pipeline.
