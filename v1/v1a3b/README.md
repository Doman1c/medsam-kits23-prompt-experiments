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
- expansion `1.20` during the decoder comparison.

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

## Stage 1 — topology audit and initial decoder sweep

Run:

```bash
bash /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_validation.sh
```

Stage 1 produced the following locked B0 topology measurements on validation:

| Metric | B0 |
|---|---:|
| Side-level positive recall | 0.9600 |
| Side-level FPR | 0.1757 |
| Left recall | 0.9420 |
| Right recall | 0.9816 |
| Multi-segment case-side rate | 0.5750 |
| False-positive segments | 62 |
| False-positive segment slices | 358 |
| Mean onset absolute error | 38.25 slices |
| Mean offset absolute error | 17.03 slices |

The topology audit supports the core V1a.3-B hypothesis: 23 of 40 validation
case-sides contain multiple predicted presence segments, and 22 case-sides
contain at least one fully detached false-positive segment. Most detached false
segments occur before the ground-truth kidney span and the left side remains the
harder side.

The initial B1/B2 grid produced zero configurations satisfying the locked recall
constraints. Its highest-recall points occurred at the edge of the search grid,
so Stage 1 was treated as search-space saturation rather than a rejection of the
span hypothesis.

## Stage 1b — widened span-support sweep

Stage 1b changed no model, geometry, split, MedSAM setting, or recall constraint.
It only widened the saturated decoder dimensions and added GT-independent
symmetric padding around the predicted span.

The sweep evaluated 3,087 configurations. Of these, 2,464 satisfied:

```text
overall recall >= 0.95
left recall    >= 0.90
right recall   >= 0.90
```

The lowest-FPR feasible B1 point was:

```text
mode              = single_span
left threshold    = 0.0035
right threshold   = 0.0400
baseline max gap  = 4
boundary pad      = 1
recall             0.9502
left/right recall  0.9275 / 0.9775
FPR                0.1193
specificity        0.8807
```

The lowest-FPR feasible B2 point was:

```text
mode               = optimal_span
left threshold     = 0.0015
right threshold    = 0.0400
max internal gap   = 3
boundary pad       = 1
recall              0.9516
left/right recall   0.9301 / 0.9775
FPR                 0.1207
specificity         0.8793
```

Both remove all detached multi-segment false-positive islands. Their mean
onset/offset absolute errors are about `11/7` slices rather than the B0
`38.25/17.03` slices.

A second important Stage-1b result is that the span decoder can also match B0's
`0.9600` recall while still reducing FPR. The minimum-FPR recall-matched points
are approximately:

| Candidate | Recall | FPR | Specificity |
|---|---:|---:|---:|
| B0 locked | 0.9600 | 0.1757 | 0.8243 |
| B1 recall-matched | 0.9600 | 0.1475 | 0.8525 |
| B2 recall-matched | 0.9600 | 0.1477 | 0.8523 |

Therefore Stage 1b passes the presence-level gate and justifies a negative-aware
MedSAM comparison.

## Stage 2 — negative-aware MedSAM

Stage 2 deliberately evaluates a compact predeclared set rather than every
Stage-1b sweep point:

```text
B0 locked baseline
B1 minimum-FPR feasible point
B1 minimum-FPR point matching B0 recall
B2 minimum-FPR feasible point
B2 minimum-FPR point matching B0 recall
```

This gives both a low-FPR operating point and a recall-matched comparison for
each decoder family. Candidate construction uses only presence metrics; MedSAM
results do not influence which configurations enter Stage 2.

If Stage 1b has already completed in the current runtime:

```bash
cd /content/medsam-kits23-prompt-experiments
git pull --ff-only origin v1a3b-span-decoder

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_stage2_medsam.sh
```

Outputs:

```text
/content/v1a1_work/v1a3b_stage2_medsam/
/content/drive/MyDrive/medsam-kits23/v1a3b_results/stage2_medsam/
```

Package the compact Stage-2 review bundle with:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/package_v1a3b_stage2.py \
  --root /content/v1a1_work/v1a3b_stage2_medsam \
  --output /content/v1a3b_stage2_review.zip
```

Stage 2 is still validation-only. A V1a.3-B frozen test is considered only if a
single decoder configuration is selected entirely from validation results.

## Test-split rule

V1a.3-B development remains validation-only. The scripts verify that prediction
rows exactly match the 20-case / 3,644-row validation manifest. No V1a.3-B test
configuration is opened during Stage 1, Stage 1b, or Stage 2.