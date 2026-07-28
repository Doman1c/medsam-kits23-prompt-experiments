# V1a.1 Frozen Test Evaluation and Error Analysis

## Status

The V1a.1 test split was evaluated once under the validation-locked configuration. The outputs in this directory are descriptive only and must not be used to retune thresholds, smoothing, gap filling, expansion, checkpoints, loss weights, or architecture variants on the test set.

## Locked configuration

```text
checkpoint      = epoch 6
weights         = EMA
prediction      = smooth
max_gap         = 0
left threshold  = 0.20
right threshold = 0.30
box expansion   = 1.20
target label    = 1
base commit     = be3e114215aac33525e846d004b66088e62e92d5
```

## Frozen test data

| Item | Count |
|---|---:|
| Test cases | 27 |
| Manifest test rows | 4,277 |
| Positive slices evaluated by MedSAM | 1,473 |

## Main MedSAM results

| Metric | Automatic | Oracle-scaled | Gap |
|---|---:|---:|---:|
| Slice mean Dice | 0.7628 | 0.8713 | 0.1085 |
| Slice median Dice | 0.9102 | 0.9415 | 0.0102 |
| Case mean Dice | 0.7950 | 0.8773 | 0.0823 |

The large mean–median difference indicates a long-tailed failure pattern: most slices are segmented well, while a smaller group of severe failures lowers the mean.

## Error-analysis findings

- Low-Dice slices (`Dice < 0.5`): **249 / 1473 (16.9%)**.
- Very-low-Dice slices (`Dice < 0.2`): **116 (7.9%)**.
- Exact left/right presence: **903 slices**, mean Dice **0.8495**.
- Extra absent-side prompt: **553 slices**, mean Dice **0.6302**.
- Among low-Dice slices, **178 (71.5%)** were assigned to absent-side false-positive prompts and **61 (24.5%)** to localization or coverage failure.
- The top five cases contributed **78%** of the positive automatic-to-oracle gap.

These categories are descriptive and can overlap mechanistically. A slice assigned to an absent-side false positive may also contain a poor box for the true side.

## Oracle-presence / automatic-box decomposition

Three frozen modes were compared:

- **A:** automatic presence + automatic smooth boxes
- **B:** ground-truth presence + automatic smooth boxes
- **C:** ground-truth presence + oracle scaled boxes

| Mode | Mean Dice | Median Dice |
|---|---:|---:|
| A | 0.7628 | 0.9102 |
| B | 0.7881 | 0.9249 |
| C | 0.8713 | 0.9415 |

The total mean oracle gap was decomposed as:

```text
presence component (B - A) = 0.0253
box component      (C - B) = 0.0832
total gap          (C - A) = 0.1085
```

Approximately **23.3%** of the signed mean gap was associated with presence decisions and **76.7%** with automatic-box localization/size. The positive-gap recovery calculation similarly attributed about **25.0%** of recoverable gap to presence correction.

## Interpretation

The principal V1a.1 limitation is a long-tailed automatic-box failure mode, particularly center, size, or coverage errors in a small set of cases. Absent-side false-positive prompts remain important, but oracle presence alone raises mean Dice only from 0.7628 to 0.7881; oracle boxes are required to reach 0.8713.

The next development version should therefore prioritize validation-only diagnosis and improvement of box regression, while treating presence specificity as a secondary objective. A useful next diagnostic is a validation-set center-versus-size decomposition before defining V1a.2.

## Archived files

### Reproduction and analysis scripts

- `scripts/run_v1a1_locked_test.sh`
- `scripts/summarize_locked_test.py`
- `scripts/analyze_v1a1_test_errors.py`
- `scripts/evaluate_oracle_presence_decomposition.py`

### Compact result summaries

- `results/test_final/error_analysis/error_analysis_summary.json`
- `results/test_final/error_analysis/failure_category_summary.csv`
- `results/test_final/error_analysis/side_status_summary.csv`
- `results/test_final/error_analysis/case_error_summary.csv`
- `results/test_final/oracle_presence_decomposition/gap_decomposition_summary.json`

Large per-slice outputs, predictions, model weights, and processed NPZ files remain outside Git. Their SHA256 values are recorded in the JSON summaries.
