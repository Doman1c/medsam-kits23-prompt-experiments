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

## Locked validation gate

V1a.4 development uses only the 20-case / 3,644-row validation split.
The recovery gate remains unchanged across Stage 1 and Stage 1b:

```text
overall recall >= 0.96
left recall    >= 0.94
right recall   >= 0.97
FPR             <= 0.15
```

No MedSAM and no test split are used before a presence decoder passes this gate.

## Stage 1 result

The Stage-1 review bundle passed its size/SHA-256 integrity checks and contained
no test result, checkpoint, NPZ, or large prediction table.

Frozen B1 validation presence:

| Metric | Frozen B1 |
|---|---:|
| Overall recall | 0.9502 |
| Left recall | 0.9275 |
| Right recall | 0.9775 |
| FPR | 0.1193 |
| Specificity | 0.8807 |
| False-negative slices | 107 |

False-negative topology showed that the error is not mainly a short local
boundary problem:

```text
83 / 107 FN slices occur before the frozen seed
24 / 107 occur after the seed
85 / 107 are left-side misses
22 / 107 are right-side misses
47 / 107 are within 16 slices of the frozen seed
```

One validation case-side (`case_00067`, left) accounts for 76 of the 107 false
negative slices. Those misses extend as far as 76 slices before the frozen seed.
This reveals a long-range split-span failure mode: the hard B1 decoder can pick a
dominant segment while discarding another substantial supported part of the same
anatomical target.

The Stage-1 sweep evaluated 738 configurations and produced zero points satisfying
the locked gate.

Best local hysteresis point under `FPR <= 0.15`:

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

R1 therefore recovers useful local boundary misses, especially on the right, but
cannot repair the dominant long-range left-side error. The original R2 search
was capped at a 16-slice satellite distance and did not improve recall over the
frozen B1 baseline.

Stage 1 therefore rejects **local-only recovery as sufficient**, not the general
satellite-recovery hypothesis. Compact findings are stored in
`results/stage1_summary.json`.

## Stage 1b — remote satellite + controlled hybrid

Stage 1b keeps the same checkpoint, B1 seed, geometry, expansion, validation
split, and recall/FPR gate.

The local R1 component is no longer swept. It is frozen to the best Stage-1
local-recovery point above. Stage 1b tests only the new missing degree of freedom:
whether one strongly supported secondary segment may be retained at a larger
z-distance.

Two controlled families are evaluated:

```text
R2b remote_satellite
  frozen B1 dominant span
  + at most one threshold-derived satellite
  + distance up to 96 slices
  + evidence-ratio and minimum-length gates

R3 hybrid
  fixed Stage-1 R1 local hysteresis
  union
  R2b remote satellite
```

The sweep does not fill the long gap between a dominant span and a remote
satellite. This avoids converting a split-span error into a very long continuous
false-positive block.

Run Stage 1b after Stage 1 in the same restored Colab runtime:

```bash
cd /content/medsam-kits23-prompt-experiments
git fetch origin
git checkout v1a4-recall-recovery
git pull --ff-only origin v1a4-recall-recovery

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/run_v1a4_stage1b.sh
```

Outputs:

```text
/content/v1a1_work/v1a4_stage1b/
/content/drive/MyDrive/medsam-kits23/v1a4_results/stage1b/
```

Package the compact review bundle with:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a4/scripts/package_v1a4_stage1b.py \
  --root /content/v1a1_work/v1a4_stage1b \
  --output /content/v1a4_stage1b_review.zip
```

Do not run MedSAM or open any test split until Stage 1b produces at least one
configuration satisfying the unchanged gate.

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

## Development rule

The already-exposed V1a.3-B secondary test is not used for V1a.4 parameter
selection. It motivates the recall-recovery question only. Stage 1 and Stage 1b
remain validation-only. Test results must not be used to change threshold,
extension, satellite distance, evidence ratio, satellite length, decoder family,
geometry, expansion, or checkpoint.
