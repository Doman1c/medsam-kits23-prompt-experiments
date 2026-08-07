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
Colab runtime, mount Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Because the repository is private, store a fresh GitHub token in Colab Secrets
under `GITHUB_TOKEN` and configure session-only Git authentication without
printing the token:

```python
from google.colab import userdata
from pathlib import Path
import os

TOKEN = userdata.get("GITHUB_TOKEN")
if not TOKEN:
    raise RuntimeError("Missing Colab Secret: GITHUB_TOKEN")

netrc = Path.home() / ".netrc"
netrc.write_text(
    "machine github.com\n"
    "login x-access-token\n"
    f"password {TOKEN}\n",
    encoding="utf-8",
)
os.chmod(netrc, 0o600)
print("GitHub authentication configured for this runtime.")
```

Then clone this branch and restore the experiment assets:

```bash
git clone -b v1a3b-span-decoder \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

bash /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/restore_v1a3b_colab.sh
```

The recovery script:

1. verifies that the repository descends from the frozen V1a.3-A merge;
2. runs the synthetic span-decoder gate;
3. restores the canonical manifest;
4. reuses or extracts processed NPZ slices and repairs manifest paths;
5. resolves V1a.3-A `epoch_018.pt` from Drive;
6. restores MedSAM and `medsam_vit_b.pth`;
7. checks only the 20-case / 3,644-row validation split;
8. writes `/content/v1a1_work/v1a3b_env.sh`.

## First V1a.3-B test

Run the topology audit and decoder sweep:

```bash
bash /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_validation.sh
```

The validation runner verifies the exact frozen V1a.3-A epoch-18 checkpoint
SHA-256 before prediction and then performs:

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
