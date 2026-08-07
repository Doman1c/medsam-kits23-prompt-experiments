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

However, the initial B1/B2 grid produced zero configurations satisfying the
locked recall constraints:

```text
overall recall >= 0.95
left recall    >= 0.90
right recall   >= 0.90
```

The highest-recall initial B1 configuration reached overall recall `0.9363`
with FPR `0.1228`. The highest-recall B2 configuration reached overall recall
`0.9386`, left/right recall `0.9011/0.9836`, and FPR `0.1296`. B2 removed all
detached multi-segment false-positive islands and reduced mean onset/offset
absolute error to approximately `13.13/6.85` slices, but it still lost too many
positive slices.

This is not treated as a final rejection of the span hypothesis because the
highest-recall B2 point occurred at the lowest threshold and largest internal
gap in the initial search grid. The first sweep therefore hit its search
boundary.

## Stage 1b — widened span-support sweep

Stage 1b changes no model, geometry, split, MedSAM setting, or recall constraint.
It only widens the two Stage-1 parameters that saturated at the grid boundary
and adds a GT-independent z-boundary padding term after the single span has been
selected.

The new search covers:

```text
lower left/right probability thresholds
single-span max gaps up to 5
optimal-span internal gaps up to 6
symmetric predicted-span boundary padding: 0,1,2,3,4,6,8 slices
```

Boundary padding is applied only to the predicted span boundaries and never
uses ground-truth onset/offset information.

If Stage 1 already ran in the current Colab runtime, update the branch and run:

```bash
cd /content/medsam-kits23-prompt-experiments
git pull origin v1a3b-span-decoder

bash /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_stage1b.sh
```

Outputs:

```text
/content/v1a1_work/v1a3b_stage1b/
/content/drive/MyDrive/medsam-kits23/v1a3b_results/stage1b/
```

Package the compact review bundle with:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/package_v1a3b_stage1b.py \
  --root /content/v1a1_work/v1a3b_stage1b \
  --output /content/v1a3b_stage1b_review.zip
```

Do not run the full negative-aware MedSAM stage until Stage 1b demonstrates at
least one B1/B2 configuration satisfying all locked recall constraints. This
avoids spending frozen-MedSAM compute on an infeasible decoder.

## Test-split rule

V1a.3-B development is validation-only. The scripts verify that the prediction
rows exactly match the validation manifest. A new frozen test evaluation is
considered only after a decoder is selected entirely on validation.
