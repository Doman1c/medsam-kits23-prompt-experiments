# V1a.4 — Recall recovery on top of frozen V1a.3-B

V1a.4 is a validation-only post-processing experiment. It does not retrain the
prompt network and does not change geometry or MedSAM. It starts from the frozen
V1a.3-B single-span decoder and tests two narrowly defined recovery mechanisms:

```text
R0 frozen B1
  single dominant span

R1 hysteresis boundary recovery
  dominant span -> lower-threshold bounded boundary extension

R2 limited satellite recovery
  dominant span -> retain at most one nearby supported secondary segment
```

The objective is to recover validation recall while keeping a substantial part
of V1a.3-B's false-positive suppression.

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

The exact experiment lock is stored in `BASELINE_LOCK.json`.

## Stage-1 gate

Stage 1 is validation-only and uses the 20-case / 3,644-row validation split.
The locked recovery gate is:

```text
overall recall >= 0.96
left recall    >= 0.94
right recall   >= 0.97
FPR             <= 0.15
```

No MedSAM and no test split are used in Stage 1.

## Fresh Colab recovery

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

For the private repository, keep a fresh GitHub token in the Colab Secret
`GITHUB_TOKEN`, then configure session-only Git authentication without printing
it:

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

Clone the V1a.4 branch and restore assets:

```bash
rm -rf /content/medsam-kits23-prompt-experiments

git clone -b v1a4-recall-recovery \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/restore_v1a4_colab.sh
```

The restore script reuses the proven V1a.3-B environment recovery, verifies that
the branch descends from the frozen V1a.3-B merge, checks the epoch-18 checkpoint,
runs the V1a.4 synthetic decoder gate, validates only the validation split, and
writes:

```text
/content/v1a1_work/v1a4_env.sh
```

## Run Stage 1

```bash
bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/run_v1a4_stage1.sh
```

The runner performs:

```text
frozen epoch-18 validation base predictions
        ↓
V1a.3-B frozen-B1 false-negative recovery audit
        ↓
R1 hysteresis sweep
R2 limited-satellite sweep
        ↓
locked recall/FPR gate
        ↓
small diverse shortlist
        ↓
prompt-level geometry check at expansion 1.20
```

Outputs:

```text
/content/v1a1_work/v1a4_stage1/
/content/drive/MyDrive/medsam-kits23/v1a4_results/stage1/
```

## Package Stage 1 for review

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/package_v1a4_stage1.py \
  --root /content/v1a1_work/v1a4_stage1 \
  --output /content/v1a4_stage1_review.zip
```

The review bundle intentionally excludes checkpoints, NPZ data, MedSAM weights,
large prediction tables, notebooks, and test results.

## Development rule

The already-exposed V1a.3-B secondary test is not used for V1a.4 parameter
selection. It motivates the problem only. All thresholds, extension lengths,
bridge gaps, and satellite rules are selected using validation. See `DESIGN.md`
for the controlled experiment rationale and stopping rules.
