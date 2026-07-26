# V0 — MedSAM Prompt Baseline

V0 contains the initial KiTS23 preprocessing, environment checks, and frozen-MedSAM prompt-ablation experiments.

## Main components

- `prepare_kits23_2d.py`: converts KiTS23 volumes into 2D slices.
- `run_v0_experiment.py`: evaluates box, point, mask-derived, and combined prompts.
- `environment_check.py`: validates the Colab environment, data paths, and checkpoint availability.

## Representative result

| Prompt mode | Mean Dice |
|---|---:|
| box_mask | 0.9460 |
| box | 0.9414 |
| all | 0.9272 |
| box_pos | 0.8475 |
| box_pos_neg | 0.6240 |

These experiments established bounding boxes as the most reliable prompt type for the later automatic prompt-generation pipeline.

Raw KiTS23 data, processed NPZ slices, and MedSAM weights are intentionally excluded from Git.
