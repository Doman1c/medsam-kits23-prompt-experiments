# V0.5 — MedSAM Robustness to Imperfect Box Prompts

This experiment evaluates how bounding-box scale, translation, and
independent corner jitter affect frozen MedSAM segmentation on KiTS23 kidney
slices.

## Dataset split

- Train cases: 100
- Validation cases: 20
- Test cases: 27, reserved for later final validation
- Formal V0.5 development analysis: train + validation
- Representative slices per case: five target-area quantiles
  (10%, 30%, 50%, 70%, 90%)

## Experiment size

- Cases: 120
- Slice-target samples: 600
- Prompt evaluations: 28200
- Mean oracle Dice: 0.8513

## Conditions

- Oracle ground-truth box
- Scale: 0.75, 0.90, 1.10, 1.25, 1.50, 2.00
- Shift: 5%, 10%, 20%, 30%, repeated five times
- Corner jitter: 5%, 10%, 20%, 30%, repeated five times

## Main metrics

- Dice and IoU
- Dice retention relative to the oracle box
- Delta Dice
- Prompt-induced failure
- Relative failure
- Severe Dice drop
- Box IoU, target coverage, center error, and area deviation
- Case-level bootstrap 95% confidence intervals

## Condition summary

| Perturbation | Level | Mean Dice | Median retention | Prompt-induced failure | Severe drop |
|---|---:|---:|---:|---:|---:|
| corner_jitter | 0.05 | 0.8475 | 0.9996 | 0.0072 | 0.0048 |
| corner_jitter | 0.10 | 0.8333 | 0.9964 | 0.0476 | 0.0604 |
| corner_jitter | 0.20 | 0.7613 | 0.9418 | 0.2588 | 0.3520 |
| corner_jitter | 0.30 | 0.6649 | 0.8329 | 0.4772 | 0.5768 |
| oracle | 0.00 | 0.8499 | 1.0000 | 0.0000 | 0.0000 |
| scale | 0.75 | 0.6409 | 0.7482 | 0.6240 | 0.7000 |
| scale | 0.90 | 0.8332 | 0.9900 | 0.0480 | 0.0560 |
| scale | 1.10 | 0.8465 | 1.0004 | 0.0100 | 0.0120 |
| scale | 1.25 | 0.8298 | 0.9967 | 0.0440 | 0.0900 |
| scale | 1.50 | 0.7073 | 0.8477 | 0.4120 | 0.6140 |
| scale | 2.00 | 0.4604 | 0.5466 | 0.7580 | 0.9280 |
| shift | 0.05 | 0.8480 | 0.9997 | 0.0056 | 0.0052 |
| shift | 0.10 | 0.8322 | 0.9955 | 0.0488 | 0.0616 |
| shift | 0.20 | 0.7477 | 0.9364 | 0.2820 | 0.3760 |
| shift | 0.30 | 0.6298 | 0.8140 | 0.4948 | 0.6136 |
| corner_jitter | 0.05 | 0.8583 | 1.0000 | 0.0040 | 0.0000 |
| corner_jitter | 0.10 | 0.8483 | 0.9953 | 0.0240 | 0.0200 |
| corner_jitter | 0.20 | 0.7810 | 0.9620 | 0.2300 | 0.2980 |
| corner_jitter | 0.30 | 0.6730 | 0.8416 | 0.4620 | 0.5580 |
| oracle | 0.00 | 0.8584 | 1.0000 | 0.0000 | 0.0000 |
| scale | 0.75 | 0.6471 | 0.7591 | 0.6000 | 0.6700 |
| scale | 0.90 | 0.8532 | 0.9908 | 0.0200 | 0.0000 |
| scale | 1.10 | 0.8581 | 1.0026 | 0.0200 | 0.0200 |
| scale | 1.25 | 0.8415 | 0.9994 | 0.0900 | 0.0800 |
| scale | 1.50 | 0.7087 | 0.8372 | 0.4300 | 0.6500 |
| scale | 2.00 | 0.4616 | 0.5306 | 0.7500 | 0.9700 |
| shift | 0.05 | 0.8586 | 1.0000 | 0.0060 | 0.0020 |
| shift | 0.10 | 0.8487 | 0.9972 | 0.0240 | 0.0240 |
| shift | 0.20 | 0.7614 | 0.9415 | 0.2560 | 0.3660 |
| shift | 0.30 | 0.6351 | 0.8208 | 0.4940 | 0.5780 |

## Reproduction

```bash
python scripts/run_v05_prompt_robustness.py \
  --processed-root /path/to/kits23_2d_all \
  --checkpoint /path/to/medsam_vit_b.pth \
  --split-file configs/splits_v1.json \
  --splits train val \
  --output-dir outputs/v05_kidney_train_val_quantile \
  --target-name kidney \
  --component-mode largest \
  --samples-per-case 5 \
  --repeats 5 \
  --seed 42 \
  --failure-dice 0.80 \
  --relative-failure-retention 0.90 \
  --severe-drop-threshold 0.10
```

```bash
python scripts/analyze_v05_results.py \
  --results outputs/v05_kidney_train_val_quantile/sample_results.csv \
  --output-dir outputs/v05_kidney_train_val_quantile \
  --failure-dice 0.80 \
  --relative-failure-retention 0.90 \
  --severe-drop-threshold 0.10 \
  --bootstrap-repeats 2000 \
  --bootstrap-seed 42
```

## Files excluded from Git

Raw KiTS23 volumes, processed NPZ slices, MedSAM checkpoints, caches, and
other large model/data files are intentionally excluded.
