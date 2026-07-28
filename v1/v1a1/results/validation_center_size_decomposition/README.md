# Validation Center-versus-Size Decomposition

This directory stores the compact result summary for the validation-only four-mode box decomposition.

| Mode | Mean Dice |
|---|---:|
| M0: automatic center + automatic size | 0.8361 |
| M1: oracle center + automatic size | 0.8510 |
| M2: automatic center + oracle size | 0.8457 |
| M3: oracle center + oracle size | 0.8708 |

Main diagnostic quantities:

```text
center gain       = 0.0149
size gain         = 0.0096
joint box gain    = 0.0347
center-size interaction = 0.0102
```

The experiment uses 20 validation cases and 1,253 positive validation slices. The frozen test set was not used.

See `../../docs/VALIDATION_CENTER_SIZE_DECOMPOSITION.md` for methods and interpretation. Large per-slice outputs remain on Drive; artifact hashes are recorded in `center_size_decomposition_summary.json`.
