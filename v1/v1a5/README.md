# V1a.5 — Geometry recovery for restored kidney slices

V1a.5 starts from the completed V1a.4 validation study. The key V1a.4 finding
was that remote-satellite presence recovery can identify additional true-positive
kidney slices without increasing validation presence FPR, but the native box on
those low-confidence slices is too poor for frozen MedSAM.

V1a.5 therefore asks:

> When case-level context says a slice should be prompted, can its spatial box be
> recovered from nearby reliable slices instead of trusting that slice's native
> center/size prediction?

No retraining is performed in Stage 1.

## Frozen upstream

```text
V1a.4 merge       = 221f7f63a3a727b96c286edf09b9ad47e84bf555
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
expansion         = 1.20
```

The V1a.4 R2b presence decoder is also frozen:

```text
satellite max distance   = 64
satellite evidence ratio = 0.10
satellite min length     = 1
max satellites           = 1
```

## Stage-1 comparison

```text
G0 = frozen V1a.3-B B1

G1 = frozen V1a.4 R2b
     + native per-slice geometry

G2 = frozen V1a.4 R2b
     + nearest reliable B1-anchor box propagation

G3 = frozen V1a.4 R2b
     + five-anchor z-axis linear box trend
```

Only side-slices newly recovered by R2b are allowed to change geometry in G2/G3.
Presence and every non-recovered box remain frozen.

Stage 1 performs geometry audit + prompt-quality evaluation only. It does not run
MedSAM and does not access any test split.

## Existing Colab runtime

If the runtime used for V1a.4 Stage 2 is still alive, update the repository and
restore the V1a.5 gate:

```bash
cd /content/medsam-kits23-prompt-experiments

git fetch origin
git checkout -B v1a5-geometry-recovery origin/v1a5-geometry-recovery

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a5/scripts/restore_v1a5_colab.sh
```

Then run Stage 1:

```bash
bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a5/scripts/run_v1a5_stage1.sh
```

The runner searches for the frozen B1 and R2b validation prediction tables in
both the current `/content` workspace and the V1a.4 Drive backups.

Outputs:

```text
/content/v1a1_work/v1a5_stage1/
/content/drive/MyDrive/medsam-kits23/v1a5_results/stage1/
```

## Fresh Colab runtime

Mount Drive and configure the private GitHub repository as before, then clone the
V1a.5 branch:

```bash
rm -rf /content/medsam-kits23-prompt-experiments

git clone -b v1a5-geometry-recovery \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a5/scripts/restore_v1a5_colab.sh
```

The restore wrapper reuses the proven V1a.4/V1a.3-B asset recovery and validates:

- frozen V1a.4 ancestry;
- epoch-18 checkpoint SHA;
- 20 validation cases / 3,644 validation rows;
- validation NPZ paths;
- MedSAM assets for later stages;
- V1a.5 synthetic geometry gate.

After restore, run the same Stage-1 command above.

## Package Stage 1

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a5/scripts/package_v1a5_stage1.py \
  --root /content/v1a1_work/v1a5_stage1 \
  --output /content/v1a5_stage1_review.zip
```

Upload `v1a5_stage1_review.zip` for review before any V1a.5 MedSAM run.

The compact review bundle excludes recommended prediction tables, per-slice
prompt details, checkpoints, NPZ data, MedSAM weights, notebooks, and test
results.

## Development rule

The V1a.3-B/V1a.4 27-case secondary split is already exposed and is not used for
V1a.5 selection. Stage-1 decisions are validation-only. See `DESIGN.md` and
`BASELINE_LOCK.json` for the frozen experiment definition.
