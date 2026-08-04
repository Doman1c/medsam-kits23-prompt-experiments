# V1a.2 validation status after follow-up checks

This branch archives validation-only development after the frozen V1a.1 baseline. The frozen test split was not used for checkpoint, threshold, smoothing, max-gap, expansion, or loss selection.

## Completed follow-up checks

1. Negative-aware frozen-MedSAM evaluation on all 3,644 validation slices, including 2,390 fully negative slices.
2. Center-size decomposition at a common expansion scale of 1.20 for V1a.1, V1a.2-A, and V1a.2-B.
3. Independent validation calibration and negative-aware checkpoint Pareto analysis for V1a.2-B epochs 4, 5, 11, and 14.

## Negative-aware result

The strongest evaluated locked automatic operating point is V1a.2-A:

| Candidate | Positive-slice Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|---:|
| V1a.2-A locked | 0.808913 | 0.766569 | **0.855560** | 0.744351 |
| V1a.2-B FPR-best | 0.791647 | 0.650309 | 0.823777 | 0.576151 |
| V1a.2-B prompt-best | **0.825550** | 0.306048 | 0.753368 | 0.033473 |
| V1a.1 locked | 0.811517 | 0.309726 | 0.711219 | 0.046444 |

The positive-slice winner is not the all-slice winner. V1a.2-B prompt-best obtains stronger segmentation where a kidney exists, but triggers false masks on nearly all negative slices. Presence behavior therefore dominates the complete automatic-system result.

## Fixed-scale geometry result

All models below use ground-truth presence and expansion 1.20:

| Model | M0 | Joint box gap | Interaction |
|---|---:|---:|---:|
| V1a.1 | 0.836142 | 0.034708 | 0.010238 |
| V1a.2-A | 0.840618 | 0.030232 | 0.012827 |
| V1a.2-B | **0.841626** | **0.029224** | **0.006936** |

V1a.2-B remains the strongest geometry model. Relative to V1a.1, it raises M0 by 0.005484, reduces the joint box gap by 0.005484, and reduces center-size interaction by 0.003302 under a controlled expansion scale.

## Checkpoint Pareto result

Among V1a.2-B epochs 4, 5, 11, and 14, epoch 14 is the best negative-aware checkpoint:

| Epoch | Positive-slice Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---:|---:|---:|---:|---:|
| 14 | 0.797241 | **0.772487** | **0.850753** | **0.759833** |
| 11 | 0.798092 | 0.741495 | 0.848519 | 0.712134 |
| 5 | 0.820649 | 0.399636 | 0.767727 | 0.179079 |
| 4 | **0.826209** | 0.306048 | 0.753368 | 0.033473 |

Later checkpoints trade some positive-slice Dice for substantially better specificity. Epoch 14 improves the V1a.2-B full-system operating point but remains below V1a.2-A locked case-volume Dice (0.850753 versus 0.855560).

## Current decision

- **Geometry conclusion:** V1a.2-B decoded-box supervision is supported and should be retained.
- **Current complete automatic operating point:** V1a.2-A locked is the best evaluated configuration.
- **V1a.2-B checkpoint:** epoch 14 is preferable to epoch 4 for negative-aware use, but is not the final winner.
- **Next development target:** presence detection/calibration and negative-slice suppression, not additional box-loss complexity.

Compact results and reproduction scripts are archived under `../followup_validation/`. Model checkpoints, MedSAM weights, processed NPZ data, and large per-slice outputs remain on Google Drive.
