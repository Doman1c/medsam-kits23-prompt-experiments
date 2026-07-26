# v1a.1 Dual-Kidney Prompt Generator

This package implements the redesigned **left/right kidney-consistent** automatic prompt generator for KiTS23 and frozen MedSAM.

## What changed from v1a

- The target is no longer the independently selected largest component on each slice.
- Raw NIfTI volumes are converted to canonical **RAS** orientation.
- Low canonical x is treated as anatomical left kidney; high x is anatomical right kidney.
- Left and right components are tracked separately along z.
- The network predicts two presence logits, two center heatmaps, local size maps, and local offset maps.
- Size is read at each predicted center instead of from one global pooled feature.
- Training adds coordinate, offset, under-size, and left-before-right ordering losses.
- Validation saves every epoch, best prompt checkpoint, best continuous checkpoint, EMA weights, and Drive backups.
- Expansion 1.10/1.15/1.20/1.25 is swept on validation after training.

## Default Google Drive layout

The code automatically searches:

```text
/content/drive/MyDrive/medsam-kits23/
    data/kits23_selected_147_cases.tar
    cache/
    v1a1_results/

/content/drive/MyDrive/medsam_vit_b.pth
```

The archive may be elsewhere under `medsam-kits23`; the largest compatible archive is discovered automatically. The MedSAM checkpoint is first searched in the root of `MyDrive`, then recursively.

## Colab start

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload `v1a1_dual_kidney_pipeline.zip` to `/content`, then:

```bash
!rm -rf /content/v1a1_dual_kidney_pipeline
!unzip -q /content/v1a1_dual_kidney_pipeline.zip -d /content
%cd /content/v1a1_dual_kidney_pipeline
!pip install -q -r requirements.txt
```

Confirm GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

## Recommended staged execution

### 1. Data, orientation, dual manifest, and sanity image

```bash
!python run_pipeline.py \
  --project-dir /content/drive/MyDrive/medsam-kits23 \
  --drive-root /content/drive/MyDrive \
  --work-root /content/v1a1_work \
  --stop-after sanity
```

Inspect:

```python
from IPython.display import Image, display
display(Image('/content/v1a1_work/manifest/dual_target_sanity.png'))
```

Important outputs:

```text
/content/v1a1_work/orientation/orientation_audit.csv
/content/v1a1_work/manifest/v1a1_dual_manifest.csv
/content/v1a1_work/manifest/dual_track_audit.csv
/content/v1a1_work/manifest/dual_target_sanity.png
```

The first preprocessing run creates this reusable Drive cache:

```text
MyDrive/medsam-kits23/cache/v1a1_processed_147_cases.tar
```

After a Colab restart, this cache avoids repeating NIfTI-to-NPZ conversion.

### 2. Deterministic dual-target overfit gate

```bash
!python scripts/overfit_dual.py \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --output-dir /content/v1a1_work/overfit \
  --positive-slices 48 \
  --negative-slices 12 \
  --steps 1200 \
  --batch-size 12 \
  --base-channels 24 \
  --expansion-scale 1.15 \
  --device cuda
```

Pass criteria:

```text
positive recall >= 0.98
median center error <= 6 px
expanded coverage >= 0.97
prompt success >= 0.85
tight box IoU >= 0.75
left/right order violations = 0
```

### 3. Full training with Drive backup after every epoch

```bash
!python scripts/train_dual.py \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --output-dir /content/v1a1_work/run \
  --backup-dir /content/drive/MyDrive/medsam-kits23/v1a1_results/run \
  --epochs 40 \
  --batch-size 32 \
  --num-workers 2 \
  --input-size 256 \
  --heatmap-size 64 \
  --input-mode 2.5d \
  --base-channels 24 \
  --lr 3e-4 \
  --warmup-epochs 3 \
  --metric-expansion-scale 1.15 \
  --early-stopping-patience 10 \
  --device cuda
```

The training directory contains:

```text
best_prompt.pt
best_continuous.pt
last.pt
checkpoints/epoch_001.pt ...
history.json
config.json
```

Use `best_continuous.pt` as the stable primary candidate and retain `best_prompt.pt` as the threshold-based candidate.

### 4. Export validation and test boxes

```bash
!python scripts/predict_dual.py \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --checkpoint /content/v1a1_work/run/best_continuous.pt \
  --output-csv /content/v1a1_work/run/val_predictions.csv \
  --split val \
  --device cuda

!python scripts/predict_dual.py \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --checkpoint /content/v1a1_work/run/best_continuous.pt \
  --output-csv /content/v1a1_work/run/test_predictions.csv \
  --split test \
  --device cuda
```

Each CSV includes raw and independently z-smoothed tight boxes for both kidneys.

### 5. Validation expansion sweep

```bash
!python scripts/sweep_expansion.py \
  --predictions /content/v1a1_work/run/val_predictions.csv \
  --output-dir /content/v1a1_work/run/val_expansion_sweep \
  --scales 1.10,1.15,1.20,1.25
```

This prompt-level sweep is preliminary. The final scale should be chosen by **MedSAM validation Dice**, not coverage alone.

### 6. Prompt evaluation

```bash
!python scripts/evaluate_dual_prompts.py \
  --predictions /content/v1a1_work/run/test_predictions.csv \
  --output-dir /content/v1a1_work/run/test_prompt_evaluation \
  --variant both \
  --expansion-scale 1.15
```

### 7. Frozen MedSAM pilot

Discover paths:

```bash
!python scripts/discover_drive_assets.py \
  --drive-root /content/drive/MyDrive \
  --project-dir /content/drive/MyDrive/medsam-kits23 \
  --output-json /content/v1a1_assets.json
```

Setup:

```bash
!python scripts/setup_medsam.py \
  --repo-dir /content/v1a1_work/MedSAM \
  --checkpoint-source /content/drive/MyDrive/medsam_vit_b.pth \
  --checkpoint-dest /content/v1a1_work/medsam_weights/medsam_vit_b.pth
```

Pilot on 100 test slices:

```bash
!python scripts/evaluate_medsam_dual.py \
  --predictions /content/v1a1_work/run/test_predictions.csv \
  --medsam-repo /content/v1a1_work/MedSAM \
  --checkpoint /content/v1a1_work/medsam_weights/medsam_vit_b.pth \
  --output-dir /content/v1a1_work/run/medsam_test_pilot \
  --expansion-scale 1.15 \
  --max-samples 100 \
  --device cuda
```

The comparison includes:

```text
oracle_tight
oracle_scaled
dual_raw
dual_smooth
```

The two MedSAM masks are unioned and compared with the complete label-1 kidney target.

## One-command full run

After inspecting the sanity image once, the pipeline can be run end-to-end:

```bash
!python run_pipeline.py \
  --project-dir /content/drive/MyDrive/medsam-kits23 \
  --drive-root /content/drive/MyDrive \
  --work-root /content/v1a1_work \
  --epochs 40 \
  --batch-size 32 \
  --num-workers 2
```

Add `--run-medsam` to include a 100-slice frozen-MedSAM pilot.

## Research safeguards

- The established 100 train / 20 validation cases are reused; the remaining 27 are test.
- Test data are not used for checkpoint, threshold, or expansion selection.
- Horizontal flipping is disabled because it would swap anatomical side identity.
- `label == 1` remains the kidney target to preserve comparability with v0/v1a.
- Both best checkpoints and every epoch are saved, preventing the earlier “epoch 4 only” recovery problem.

---

## Post-training calibration after early stopping

Use `POST_TRAINING_CALIBRATION.md` for the validation-only workflow added after the dual model early-stopped. The workflow calibrates independent left/right presence thresholds, z-axis smoothing, gap filling, expansion scale, and frozen-MedSAM validation before one-time test evaluation.
