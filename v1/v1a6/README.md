# V1a.6 — Satellite-local geometry consensus

V1a.6 starts from the completed V1a.5 study. V1a.5 showed that geometry copied
or extrapolated from the distant frozen-B1 span is not useful for the remote
R2b satellite: the nearest B1 anchors are 49–73 slices away.

V1a.6 therefore asks a narrower question:

> Can the recovered satellite reconstruct a stable MedSAM box from its own
> native per-slice geometry predictions, without borrowing geometry from the
> distant dominant span?

No retraining is performed in Stage 1.

## Frozen upstream

```text
V1a.5 merge       = af2b92b96c0fecaecbfa21038b9d3aae3de27378
checkpoint        = V1a.3-A epoch 18 EMA
checkpoint SHA256 = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
expansion         = 1.20

R2b presence:
satellite max distance   = 64
satellite evidence ratio = 0.10
satellite min length     = 1
max satellites           = 1
```

## Stage-1 candidates

```text
G1 = R2b + native per-slice geometry          [reference]
G4 = satellite-wide median geometry
G5 = within-satellite rolling median, window 5
G6 = satellite-local robust linear geometry trend
```

All G4/G5/G6 methods use only the native predicted boxes inside each recovered
satellite segment. Ground truth is used only for validation evaluation, never to
construct a candidate box.

Only R2b-recovered side-slices may change geometry. R2b presence and every
non-recovered box remain frozen.

## Stage-1 gate

A method may advance to negative-aware MedSAM only if, relative to G1, it:

1. improves recovered-slice mean tight-box IoU; and
2. lowers recovered-slice median normalized center error;
3. while passing all presence/non-recovered-geometry invariance checks.

No MedSAM and no test/secondary-test split are used before this gate.

## Existing Colab runtime

```bash
cd /content/medsam-kits23-prompt-experiments

git fetch origin
git checkout -B \
  v1a6-satellite-local-geometry \
  origin/v1a6-satellite-local-geometry

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a6/scripts/restore_v1a6_colab.sh

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a6/scripts/run_v1a6_stage1.sh
```

Outputs:

```text
/content/v1a1_work/v1a6_stage1/
/content/drive/MyDrive/medsam-kits23/v1a6_results/stage1/
```

Package the compact review bundle:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a6/scripts/package_v1a6_stage1.py \
  --root /content/v1a1_work/v1a6_stage1 \
  --output /content/v1a6_stage1_review.zip
```

Upload `v1a6_stage1_review.zip` before any MedSAM run.

## Development rule

The already-exposed 27-case split is not used for V1a.6 selection. Stage 1 is
validation-only. The median, rolling-window size, robust-inlier rule, and trend
form are fixed before Stage 1 and are not swept after seeing geometry results.
