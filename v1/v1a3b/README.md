# V1a.3-B — Case-level kidney-span decoder

V1a.3-B is a controlled post-processing experiment built on the frozen
V1a.3-A epoch-18 EMA prompt network. It tests whether explicit z-axis continuity
can reduce residual kidney-presence false positives without retraining the prompt
network or changing box geometry.

## Frozen upstream

Unchanged throughout V1a.3-B:

- V1a.3-A epoch-18 EMA checkpoint;
- 2.5D input and bilateral presence/center/size heads;
- V1a.2-B geometry design and loss;
- frozen MedSAM;
- KiTS23 split;
- MedSAM expansion `1.20`.

Only case-level presence decoding changes.

The upstream checkpoint SHA-256 is:

```text
f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2
```

## Decoder families

```text
B0 locked baseline
  3-slice median smoothing
  -> threshold
  -> isolated-positive removal
  -> local gap fill

B1 single span
  B0 candidate segments
  -> retain the single strongest segment
  -> optional symmetric z-boundary padding

B2 optimal span
  smoothed probabilities
  -> threshold-relative log evidence
  -> one optimal contiguous span with bounded internal low-confidence gaps
  -> optional symmetric z-boundary padding
```

## Environment recovery

In a fresh Colab runtime, mount Drive and configure private GitHub access through
the `GITHUB_TOKEN` Colab Secret. Then:

```bash
git clone -b v1a3b-span-decoder \
  https://github.com/Doman1c/medsam-kits23-prompt-experiments.git \
  /content/medsam-kits23-prompt-experiments

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/restore_v1a3b_colab.sh
```

The restore script verifies the frozen V1a.3-A base, restores/repairs the
manifest and processed NPZ paths, resolves epoch 18 and MedSAM assets, checks
the 20-case / 3,644-row validation split, and writes:

```text
/content/v1a1_work/v1a3b_env.sh
```

## Stage 1 — topology audit

Locked B0 validation topology:

| Metric | B0 |
|---|---:|
| Side-level positive recall | 0.9600 |
| Side-level FPR | 0.1757 |
| Left recall | 0.9420 |
| Right recall | 0.9816 |
| Multi-segment case-side rate | 0.5750 |
| Detached false-positive segments | 62 |
| Detached false-positive slices | 358 |
| Mean onset absolute error | 38.25 slices |
| Mean offset absolute error | 17.03 slices |

The audit showed that many residual errors were detached z-axis activation
islands rather than only small local boundary errors.

The first B1/B2 sweep had zero configurations meeting the locked recall
constraints because its highest-recall points landed on the search boundary.

## Stage 1b — widened support sweep

Stage 1b widened only validation-time decoder support. It did not change the
model, geometry, MedSAM, split, or recall constraints.

Locked constraints:

```text
overall recall >= 0.95
left recall    >= 0.90
right recall   >= 0.90
```

The sweep evaluated 3,087 configurations; 2,464 were feasible.

Lowest-FPR feasible points:

| Candidate | Recall | FPR | Specificity |
|---|---:|---:|---:|
| B0 locked | 0.9600 | 0.1757 | 0.8243 |
| B1 low-FPR | 0.9502 | **0.1193** | **0.8807** |
| B2 low-FPR | 0.9516 | 0.1207 | 0.8793 |

Recall-matched points also improved over B0:

| Candidate | Recall | FPR | Specificity |
|---|---:|---:|---:|
| B0 locked | 0.9600 | 0.1757 | 0.8243 |
| B1 recall-matched | 0.9600 | 0.1475 | 0.8525 |
| B2 recall-matched | 0.9600 | 0.1477 | 0.8523 |

B1/B2 removed detached multi-segment islands and reduced onset/offset absolute
error to about `11/7` slices.

## Stage 2 — negative-aware frozen-MedSAM validation

Five candidates were fixed from presence metrics before MedSAM evaluation:

```text
B0 locked baseline
B1 low-FPR
B1 recall-matched
B2 low-FPR
B2 recall-matched
```

Validation results:

| Candidate | All-slice Dice | Positive Dice | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|
| **B1 low-FPR** | **0.8356** | 0.8102 | **0.8490** | **0.8548** |
| B2 low-FPR | 0.8335 | 0.8120 | 0.8448 | 0.8544 |
| B1 recall-matched | 0.8211 | **0.8167** | 0.8234 | 0.8500 |
| B2 recall-matched | 0.8211 | **0.8167** | 0.8234 | 0.8499 |
| B0 / V1a.3-A | 0.7760 | 0.8067 | 0.7598 | 0.8479 |

B1 low-FPR is the validation winner under the predeclared ranking rule:
case-volume Dice, then all-slice Dice, then lower negative false-positive mask
rate.

Compared with B0, B1 low-FPR changes:

```text
all-slice Dice          +0.0597
positive-slice Dice     +0.0035
negative specificity    +0.0891
case-volume Dice        +0.0068
```

The paired 20-case case-volume improvement is not claimed as statistically
significant; the bootstrap confidence interval crosses zero.

## Frozen V1a.3-B configuration

`FROZEN_LOCK.json` records the validation decision before the secondary test is
run.

```text
upstream checkpoint = V1a.3-A epoch 18
weights             = EMA

decoder             = single_span
left threshold      = 0.0035
right threshold     = 0.0400
baseline max gap    = 4
boundary pad        = 1

expansion scale     = 1.20
target label        = kidney (1)
```

No parameter may be changed after the secondary-test results are opened.

## Locked secondary test

The original 27-case test split was already exposed during V1a.3-A. Therefore
V1a.3-B uses it only as a **locked secondary evaluation**, not as a pristine
first-look test.

If the current Colab environment is already restored:

```bash
cd /content/medsam-kits23-prompt-experiments
git pull --ff-only origin v1a3b-span-decoder

bash \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/run_v1a3b_secondary_test.sh
```

The runner:

1. verifies the exact epoch-18 checkpoint SHA;
2. verifies `FROZEN_LOCK.json`;
3. reuses the already-generated V1a.3-A test base predictions when available;
4. otherwise regenerates the same epoch-18 EMA base predictions;
5. materializes only B0 and frozen B1;
6. computes prompt metrics;
7. runs both candidates together through negative-aware MedSAM;
8. requires B0 to reproduce the published V1a.3-A secondary-test baseline;
9. writes a B1-minus-B0 comparison;
10. backs up results to Drive.

Outputs:

```text
/content/v1a1_work/v1a3b_secondary_test/
/content/drive/MyDrive/medsam-kits23/v1a3b_results/secondary_test/
```

Package the compact review bundle with:

```bash
python \
  /content/medsam-kits23-prompt-experiments/v1/v1a3b/scripts/package_v1a3b_secondary_test.py \
  --root /content/v1a1_work/v1a3b_secondary_test \
  --output /content/v1a3b_secondary_test_review.zip
```

## Test policy

V1a.3-B development and selection are complete before this secondary test. The
secondary test may be used only to report the behavior of the frozen B1
configuration relative to the already-known B0 baseline. It must not be used to:

- change the checkpoint;
- change thresholds;
- change gap or boundary padding;
- change expansion;
- switch back to B2;
- select another decoder configuration.

For a publication-quality untouched estimate, a new holdout from previously
unused KiTS23 cases should be created later.
