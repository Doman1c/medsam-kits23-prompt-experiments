# Reproduce the completed V1a.2 follow-up validation

The three planned follow-up validation tests are complete. This branch contains the source code and compact result summaries. Large assets remain in Google Drive and must not be committed.

## Completed tests

1. Negative-aware evaluation over all validation slices.
2. Fixed-expansion (`1.20`) center-size decomposition for V1a.1, V1a.2-A, and V1a.2-B.
3. V1a.2-B checkpoint-Pareto evaluation for epochs 4, 5, 11, and 14.

The archived conclusions are in `V1A2_VALIDATION_STATUS.md`. Reproduction tools and compact outputs are under `../followup_validation/`.

## Required Drive assets for reproduction

```text
/content/drive/MyDrive/medsam_vit_b.pth
/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2a_run_stable/
/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2b_run/
/content/drive/MyDrive/medsam-kits23/v1a2_results/reproducibility/v1a1_dual_manifest.csv
/content/drive/MyDrive/medsam-kits23/cache/v1a1_processed_147_cases.tar
```

The V1a.2-B checkpoint directory must retain:

```text
best_continuous.pt
best_prompt.pt
last.pt
checkpoints/epoch_004.pt
checkpoints/epoch_005.pt
checkpoints/epoch_011.pt
checkpoints/epoch_014.pt
```

## Fresh Colab reproduction

1. Mount Google Drive.
2. Clone branch `v1a2b-decoded-box-validation`.
3. Restore the processed cache so manifest `npz_path` entries resolve.
4. Restore V1a.2-A and V1a.2-B run directories from Drive.
5. Clone MedSAM and use the Drive checkpoint.
6. Run the scripts under `v1/v1a2b/followup_validation/tools/` only when reproducing the completed validation.

Training, overfit gates, smoke tests, and the earlier presence-Pareto experiments do not need to be repeated. The frozen test split must remain untouched unless a later protocol explicitly freezes a final model and authorizes the one-time test evaluation.
