# v1a.1 Post-training calibration

This add-on implements the validation-only plan after early stopping:

1. Export validation predictions from `best_continuous.pt`, `best_prompt.pt`, and optionally `epoch_005.pt`.
2. Sweep independent left/right presence thresholds.
3. Compare raw prediction with 3-slice smoothing and gap filling of 0, 1, or 2 slices.
4. Require overall recall >= 0.95 and each-side recall >= 0.90 by default.
5. Sweep box expansion scales 1.10–1.25 using prompt metrics.
6. Select the final expansion and checkpoint using frozen MedSAM validation Dice.
7. Lock all parameters before running the test split.

## Important

- Do not use `last.pt`; epoch 17 had a non-finite training heatmap loss.
- Thresholds, smoothing and expansion are selected on validation only.
- Test is evaluated once after the validation configuration is frozen.

## Colab commands

Assume:

```text
PROJECT=/content/v1a1_dual_kidney_pipeline_calibrated
MANIFEST=/content/v1a1_work/manifest/v1a1_dual_manifest.csv
RUN_DRIVE=/content/drive/MyDrive/medsam-kits23/v1a1_results/run
CAL=/content/v1a1_work/calibration
```

### 1. Export validation predictions

```bash
python scripts/predict_dual.py \
  --manifest "$MANIFEST" \
  --checkpoint "$RUN_DRIVE/best_continuous.pt" \
  --output-csv "$CAL/epoch6/val_base_predictions.csv" \
  --split val --weights ema --batch-size 32 --device cuda

python scripts/predict_dual.py \
  --manifest "$MANIFEST" \
  --checkpoint "$RUN_DRIVE/best_prompt.pt" \
  --output-csv "$CAL/epoch7/val_base_predictions.csv" \
  --split val --weights ema --batch-size 32 --device cuda
```

Optional high-recall candidate:

```bash
python scripts/predict_dual.py \
  --manifest "$MANIFEST" \
  --checkpoint "$RUN_DRIVE/checkpoints/epoch_005.pt" \
  --output-csv "$CAL/epoch5/val_base_predictions.csv" \
  --split val --weights ema --batch-size 32 --device cuda
```

### 2. Calibrate thresholds and smoothing

Run once per candidate:

```bash
python scripts/calibrate_presence_smoothing.py \
  --predictions "$CAL/epoch6/val_base_predictions.csv" \
  --output-dir "$CAL/epoch6/presence_smoothing" \
  --left-thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55 \
  --right-thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55 \
  --max-gaps 0,1,2 \
  --expansion-scale 1.15 \
  --min-overall-recall 0.95 \
  --min-side-recall 0.90
```

Repeat for `epoch7` and optionally `epoch5` by changing paths.

Outputs:

- `presence_smoothing_sweep.csv`
- `top20_presence_smoothing.csv`
- `recommended_calibration.json`
- `recommended_predictions.csv`

### 3. Prompt-metric expansion sweep

```bash
python scripts/sweep_expansion_calibrated.py \
  --predictions "$CAL/epoch6/presence_smoothing/recommended_predictions.csv" \
  --calibration-json "$CAL/epoch6/presence_smoothing/recommended_calibration.json" \
  --output-dir "$CAL/epoch6/expansion" \
  --scales 1.10,1.15,1.20,1.25
```

### 4. Setup MedSAM

```bash
python scripts/setup_medsam.py \
  --repo-dir /content/v1a1_work/MedSAM \
  --checkpoint-source /content/drive/MyDrive/medsam_vit_b.pth \
  --checkpoint-dest /content/v1a1_work/medsam_weights/medsam_vit_b.pth
```

### 5. MedSAM pilot on 100 validation slices

```bash
python scripts/medsam_validation_sweep.py \
  --predictions "$CAL/epoch6/presence_smoothing/recommended_predictions.csv" \
  --calibration-json "$CAL/epoch6/presence_smoothing/recommended_calibration.json" \
  --medsam-repo /content/v1a1_work/MedSAM \
  --checkpoint /content/v1a1_work/medsam_weights/medsam_vit_b.pth \
  --output-dir "$CAL/epoch6/medsam_pilot" \
  --scales 1.10,1.15,1.20,1.25 \
  --max-samples 100 \
  --device cuda
```

Remove `--max-samples 100` for full validation. Repeat for epoch 7 and optional epoch 5.

### 6. Compare checkpoint candidates

```bash
python scripts/compare_calibrated_candidates.py \
  --candidate epoch6="$CAL/epoch6/medsam_full/medsam_validation_summary.json" \
  --candidate epoch7="$CAL/epoch7/medsam_full/medsam_validation_summary.json" \
  --candidate epoch5="$CAL/epoch5/medsam_full/medsam_validation_summary.json" \
  --output-dir "$CAL/comparison"
```

After the validation winner is selected, keep its checkpoint, thresholds, smoothing mode/gap and expansion scale unchanged for test evaluation.
