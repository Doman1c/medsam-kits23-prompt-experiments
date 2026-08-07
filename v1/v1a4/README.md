# V1a.4 — Recall recovery on top of frozen V1a.3-B

V1a.4 is a validation-only post-processing experiment. It does not retrain the
prompt network and does not change geometry or MedSAM. It starts from the frozen
V1a.3-B single-span decoder and tests narrowly defined recovery mechanisms.

```text
R0 frozen B1
  single dominant span

R1 local hysteresis boundary recovery
  dominant span -> lower-threshold bounded boundary extension

R2 local satellite recovery
  dominant span -> retain at most one nearby supported secondary segment

R2b remote satellite recovery
  dominant span -> retain at most one strongly supported farther segment

R3 hybrid
  fixed Stage-1 local hysteresis + R2b remote satellite
```

## Frozen baseline

```text
upstream checkpoint = V1a.3-A epoch 18 EMA
checkpoint SHA256   = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2

V1a.3-B seed:
mode                = single_span
left threshold      = 0.0035
right threshold     = 0.0400
baseline max gap    = 4
boundary pad        = 1
expansion scale     = 1.20
```

The exact baseline and development rules are stored in `BASELINE_LOCK.json`.

## Locked validation gate

V1a.4 uses only the 20-case / 3,644-row validation split for decoder selection.
The gate is unchanged throughout development:

```text
overall recall >= 0.96
left recall    >= 0.94
right recall   >= 0.97
FPR             <= 0.15
```

No MedSAM or test access is allowed before a presence decoder passes this gate.

## Stage 1 result

Frozen B1 validation presence:

| Metric | Frozen B1 |
|---|---:|
| Overall recall | 0.9502 |
| Left recall | 0.9275 |
| Right recall | 0.9775 |
| FPR | 0.1193 |
| Specificity | 0.8807 |
| False-negative slices | 107 |

The false-negative audit showed a long-range split-span failure mode:

```text
83 / 107 FN slices occur before the frozen seed
24 / 107 occur after the seed
85 / 107 are left-side misses
22 / 107 are right-side misses
47 / 107 are within 16 slices of the frozen seed
```

One validation case-side (`case_00067`, left) accounts for 76 FN slices and the
misses extend up to 76 slices before the dominant seed.

The best local hysteresis point under `FPR <= 0.15` was:

```text
left recovery threshold  = 0.0005
right recovery threshold = 0.0025
max extension            = 6
bridge gap               = 2

overall recall = 0.9567
left recall    = 0.9318
right recall   = 0.9867
FPR            = 0.1341
```

Stage 1 swept 738 configurations and none met the full gate. Nearby satellite
recovery capped at 16 slices did not improve recall. Local-only recovery was
therefore rejected as sufficient.

## Stage 1b result

Stage 1b widened only the satellite support range and kept the Stage-1 local
hysteresis rule frozen. It evaluated 360 validation configurations:

```text
feasible total            = 40
remote_satellite feasible = 20
hybrid feasible           = 20
```

The recovery appears only once the maximum satellite distance reaches 64 slices;
searches capped at 48 slices behave like the corresponding references. This
matches the Stage-1 topology audit.

Two Stage-2 finalists are fixed before MedSAM:

| Candidate | Recall | Left recall | Right recall | FPR | Prompt success | Continuous |
|---|---:|---:|---:|---:|---:|---:|
| Frozen B1 | 0.9502 | 0.9275 | 0.9775 | 0.1193 | 0.8233 | 0.7787 |
| Fixed local hysteresis | 0.9567 | 0.9318 | 0.9867 | 0.1341 | 0.8274 | 0.7817 |
| **R2b remote satellite** | **0.9619** | **0.9488** | 0.9775 | **0.1193** | 0.8233 | 0.7819 |
| **R3 hybrid** | **0.9684** | **0.9531** | **0.9867** | 0.1341 | **0.8274** | **0.7849** |

R2b is especially important because it crosses the recall gate while preserving
frozen-B1 FPR exactly on validation. R3 gives the highest recall and prompt-level
continuous score while retaining the fixed local-hysteresis FPR.

Compact Stage-1/Stage-1b findings are stored under `results/`. The exact
four-candidate Stage-2 set is frozen in `STAGE2_LOCK.json` before MedSAM.

## Stage 2 — negative-aware MedSAM validation

Stage 2 compares exactly four predeclared candidates:

```text
1. v1a3b_frozen_b1
2. stage1_fixed_local_hysteresis
3. r2b_remote_satellite_1
4. r3_hybrid_1
```

No MedSAM result is used to select these inputs.

If the current Colab runtime is already restored and Stage 1b has completed:

```bash
cd /content/medsam-kits23-prompt-experiments
git fetch origin
git checkout v1a4-recall-recovery
git pull --ff-only origin v1a4-recall-recovery

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/run_v1a4_stage2_medsam.sh
```

Outputs:

```text
/content/v1a1_work/v1a4_stage2_medsam/
/content/drive/MyDrive/medsam-kits23/v1a4_results/stage2_medsam/
```

Package the compact review bundle with:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/package_v1a4_stage2.py \
  --root /content/v1a1_work/v1a4_stage2_medsam \
  --output /content/v1a4_stage2_review.zip
```

Do not open any test split after Stage 2. First freeze the validation winner and
record the final V1a.4 interpretation.

## Fresh Colab recovery

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

For the private repository, use the Colab Secret `GITHUB_TOKEN` for session-only
Git authentication, then clone and restore:

```bash
rm -rf /content/medsam-kits23-prompt-experiments

git clone -b v1a4-recall-recovery \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/restore_v1a4_colab.sh
```

The restore script validates the frozen V1a.3-B ancestry, epoch-18 checkpoint,
20-case validation split, processed data paths, MedSAM assets, and synthetic
decoder gates.

## Development rule

The already-exposed V1a.3-B secondary test motivates the recall-recovery question
only. It is not used for V1a.4 parameter selection. No test result may be used to
change threshold, extension, satellite distance, evidence ratio, minimum length,
decoder family, geometry, expansion, or checkpoint.
