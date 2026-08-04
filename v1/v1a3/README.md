# V1a.3-A — Presence repair by targeted sampling

V1a.3-A starts after the completed V1a.2 follow-up validation.

The previous stage established two separate conclusions:

1. **V1a.2-B is the strongest box-geometry design.**
2. **Presence decisions on fully negative and kidney-boundary slices remain the
   main limitation of the complete automatic system.**

V1a.3-A therefore keeps the V1a.2-B architecture and optimized loss unchanged.
Its single intended modeling change is the training sampler.

## Controlled change

Unchanged from V1a.2-B:

- 2.5D input;
- bilateral presence heads;
- heatmap center prediction;
- predicted-center size supervision;
- normalized decoded-corner SmoothL1;
- optimizer, EMA, augmentation, model capacity, and data split.

Changed in V1a.3-A:

- mined hard-negative oversampling;
- kidney-span boundary-negative oversampling;
- near-negative retention;
- limited far-negative sampling;
- unilateral-positive oversampling to strengthen the absent contralateral side.

## Sampling categories

Each training row is assigned exactly one category:

```text
positive_bilateral
positive_unilateral
negative_mined
negative_boundary
negative_near
negative_far
```

Default expected composition:

```text
fully negative rows = 45% of sampled rows

within negative rows:
  mined hard negatives    = 45%
  boundary negatives      = 35%
  manifest near-negatives = 15%
  far negatives           = 5%
```

Shares are automatically renormalized when a category is unavailable.

## Checkpoint policy

Training saves three prompt-network proxy checkpoints:

```text
best_presence.pt
best_geometry.pt
best_joint.pt
```

These are not final selections. They must later undergo validation-only
calibration, frozen-MedSAM evaluation, and all-slice negative-aware comparison.

Final candidate ranking remains:

1. negative-aware case-volume Dice;
2. all-slice Dice;
3. lower negative false-positive mask rate;

subject to locked recall constraints.

## Colab workflow

Clone the branch:

```bash
!git clone \
  -b v1a3-presence-repair \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments
```

Run the full V1a.3-A preparation and training pipeline:

```bash
!bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a3/scripts/run_v1a3a_colab.sh
```

The helper performs:

```text
V1a.2-B epoch-14 predictions on train split
-> presence error audit
-> hard-negative mining
-> deterministic sampler gate
-> full V1a.3-A training
-> Google Drive checkpoint backup
```

Default paths:

```text
repository:
  /content/medsam-kits23-prompt-experiments

manifest:
  /content/v1a1_work/manifest/v1a1_dual_manifest.csv

working output:
  /content/v1a1_work/v1a3a_run

Drive backup:
  /content/drive/MyDrive/medsam-kits23/v1a3_results/v1a3a_run
```

All paths can be overridden with environment variables.

## Individual stages

Presence audit:

```bash
python v1/v1a3/scripts/audit_presence_errors.py \
  --repo-root /content/medsam-kits23-prompt-experiments \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --predictions /content/v1a1_work/v1a3a_presence_audit/train_predictions.csv \
  --output-dir /content/v1a1_work/v1a3a_presence_audit/train \
  --split train
```

Sampler gate:

```bash
python v1/v1a3/scripts/test_v1a3_sampler.py \
  --repo-root /content/medsam-kits23-prompt-experiments \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --hard-negative-csv \
    /content/v1a1_work/v1a3a_presence_audit/train/hard_negative_pool.csv \
  --output-dir /content/v1a1_work/v1a3a_presence_audit/sampler_gate
```

Training:

```bash
python v1/v1a3/scripts/train_v1a3a.py \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --hard-negative-csv \
    /content/v1a1_work/v1a3a_presence_audit/train/hard_negative_pool.csv \
  --output-dir /content/v1a1_work/v1a3a_run \
  --backup-dir \
    /content/drive/MyDrive/medsam-kits23/v1a3_results/v1a3a_run \
  --device cuda
```

## Required gates

Stop before the full run when:

- the hard-negative pool is unexpectedly empty;
- the realized negative fraction differs materially from the requested fraction;
- mined or boundary categories receive zero samples while available;
- any training loss becomes non-finite;
- geometry metrics collapse relative to V1a.2-B.

## Test-split rule

V1a.3 development remains validation-only. The frozen test split is not used for
sampler, checkpoint, threshold, smoothing, expansion, or loss selection.
