# V1a.2 follow-up validation report

All analyses in this report use the validation split. The frozen test split was not used.

## 1. Negative-aware all-slice evaluation

The evaluation contains 3,644 validation slices: 1,254 slices with a non-empty kidney mask and 2,390 fully negative slices.

| Candidate | Positive-slice Dice | All-slice Dice | Case-volume Dice | Negative specificity | Negative prompt rate |
|---|---:|---:|---:|---:|---:|
| V1a.2-A locked | 0.808913 | 0.766569 | **0.855560** | 0.744351 | 0.272385 |
| V1a.2-B FPR-best | 0.791647 | 0.650309 | 0.823777 | 0.576151 | 0.434728 |
| V1a.2-B continuous-best | 0.801927 | 0.652474 | 0.816242 | 0.574059 | 0.434728 |
| V1a.2-B target-recall | 0.798164 | 0.540312 | 0.790675 | 0.405021 | 0.608368 |
| V1a.2-B prompt-best | **0.825550** | 0.306048 | 0.753368 | 0.033473 | 0.982008 |
| V1a.1 locked | 0.811517 | 0.309726 | 0.711219 | 0.046444 | 0.962343 |

The positive-slice winner is not the complete automatic-system winner. V1a.2-B prompt-best improves segmentation on slices containing kidneys but creates false masks on nearly all fully negative slices. V1a.2-A locked has the highest case-volume Dice and is the strongest evaluated automatic operating point.

## 2. Fixed expansion 1.20 center-size decomposition

All modes use ground-truth presence, the same 1,254 positive validation slices, the same MedSAM checkpoint, and expansion 1.20.

| Model | M0 | M1 | M2 | M3 | Joint box gap | Interaction |
|---|---:|---:|---:|---:|---:|---:|
| V1a.1 | 0.836142 | 0.851040 | 0.845715 | 0.870850 | 0.034708 | 0.010238 |
| V1a.2-A | 0.840618 | 0.853665 | 0.844976 | 0.870850 | 0.030232 | 0.012827 |
| V1a.2-B | **0.841626** | **0.855401** | **0.850139** | 0.870850 | **0.029224** | **0.006936** |

V1a.2-B is the strongest controlled geometry model. Relative to V1a.1, M0 improves by 0.005484, the joint box gap decreases by 0.005484, and center-size interaction decreases by 0.003302. This supports retaining decoded-box supervision.

## 3. V1a.2-B checkpoint Pareto

Each checkpoint was independently calibrated on validation and then evaluated with the same negative-aware protocol.

| Epoch | Positive-slice Dice | All-slice Dice | Case-volume Dice | Negative specificity | Side-level presence FPR |
|---:|---:|---:|---:|---:|---:|
| 14 | 0.797241 | **0.772487** | **0.850753** | **0.759833** | 0.199105 |
| 11 | 0.798092 | 0.741495 | 0.848519 | 0.712134 | 0.236279 |
| 5 | 0.820649 | 0.399636 | 0.767727 | 0.179079 | 0.673219 |
| 4 | **0.826209** | 0.306048 | 0.753368 | 0.033473 | 0.805956 |

Epoch 14 is the best negative-aware V1a.2-B checkpoint. Later training trades some positive-slice Dice for much better specificity. Epoch 14 nevertheless remains slightly below V1a.2-A locked case-volume Dice: 0.850753 versus 0.855560.

## Final decision

- Retain V1a.2-B decoded-box supervision as the leading geometry design.
- Keep V1a.2-A locked as the best currently evaluated complete automatic operating point.
- Prefer epoch 14 over epoch 4 when evaluating V1a.2-B as an automatic model.
- Direct the next experiment toward presence detection, negative-slice suppression, and presence-aware model selection rather than adding further box-loss complexity.

Large per-slice tables, checkpoints, MedSAM weights, and processed medical images remain on Google Drive. The Git archive contains the source code, locked configurations, compact comparison tables, and final machine-readable summary.
