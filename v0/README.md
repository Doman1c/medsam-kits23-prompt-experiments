# V0 — MedSAM Prompt Baseline

V0 evaluates frozen MedSAM using manually derived and oracle spatial prompts on KiTS23 kidney CT slices.

Main scripts:

- `prepare_kits23_2d.py`
- `run_v0_experiment.py`
- `environment_check.py`

Representative random-100 results:

| Prompt mode | Mean Dice |
|---|---:|
| box_mask | 0.9460 |
| box | 0.9414 |
| all | 0.9272 |
| box_pos | 0.8475 |
| box_pos_neg | 0.6240 |
