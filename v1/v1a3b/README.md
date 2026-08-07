# V1a.3-B — Case-level kidney-span decoder

V1a.3-B is a validation-only post-processing experiment built on the frozen
V1a.3-A epoch-18 prompt network. It tests whether explicit z-axis continuity can
repair residual presence false positives without retraining the prompt network
or changing box geometry.

## What changes

Unchanged:

- V1a.3-A epoch-18 EMA checkpoint;
- 2.5D input and bilateral heads;
- V1a.2-B center/size/decoded-box geometry;
- frozen MedSAM;
- KiTS23 case split;
- expansion `1.20` during the first decoder comparison.

Changed:

- case-level presence decoding only.

The three controlled modes are:

```text
B0 locked baseline
  3-slice median smoothing -> threshold -> isolated-positive removal -> gap fill

B1 single span
  B0 candidate segments -> retain the single strongest segment

B2 optimal span
  smoothed probabilities -> threshold-relative log evidence
  -> one contiguous span with bounded internal low-confidence gap
```

## Environment recovery

The scripts intentionally replace the temporary recovery notebook. In a fresh
Colab runtime:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone this branch, then run:

```bash
git clone -b v1a3b-span-decoder \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

bash /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/restore_v1a3b_colab.sh
```

The recovery script:

1. verifies that the repository descends from the frozen V1a.3-A merge;
2. restores the canonical manifest;
3. reuses or extracts processed NPZ slices and repairs manifest paths;
4. resolves V1a.3-A `epoch_018.pt` from Drive;
5. restores MedSAM and `medsam_vit_b.pth`;
6. checks only the 20-case / 3,644-row validation split;
7. writes `/content/v1a1_work/v1a3b_env.sh`.

## First V1a.3-B test

Run the topology audit and decoder sweep:

```bash
bash /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_validation.sh
```

This performs:

```text
V1a.3-A epoch-18 validation predictions
-> locked B0 topology audit
-> B1/B2 validation-only decoder sweep
-> recall-constrained shortlist
-> prompt-level comparison
-> backup to Google Drive
```

By default `RUN_MEDSAM=0`, so the first pass does not spend time on a full
negative-aware MedSAM run. After reviewing the shortlist, the same script can be
run with:

```bash
RUN_MEDSAM=1 bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_validation.sh
```

Outputs:

```text
/content/v1a1_work/v1a3b_validation/
/content/drive/MyDrive/medsam-kits23/v1a3b_results/validation/
```

## Test-split rule

V1a.3-B development is validation-only. The scripts verify that the prediction
rows exactly match the validation manifest and do not expose a `--split test`
option. A new frozen test evaluation is considered only after a decoder is
selected entirely on validation.
