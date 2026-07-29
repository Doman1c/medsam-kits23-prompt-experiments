# Resume the three follow-up validation tests in a fresh Colab

The Git branch contains all source code and compact result summaries. Large
assets remain in Google Drive and must not be committed.

## Required Drive assets

```text
/content/drive/MyDrive/medsam_vit_b.pth
/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2a_run_stable/
/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2b_run/
/content/drive/MyDrive/medsam-kits23/v1a2_results/reproducibility/v1a1_dual_manifest.csv
/content/drive/MyDrive/medsam-kits23/cache/v1a1_processed_147_cases.tar
```

The V1a.2-B checkpoint directory must retain at least:

```text
best_continuous.pt
best_prompt.pt
last.pt
checkpoints/epoch_004.pt
checkpoints/epoch_005.pt
checkpoints/epoch_011.pt
checkpoints/epoch_014.pt
```

## Fresh Colab setup

1. Mount Google Drive.
2. Clone this branch.
3. Restore the processed-cache archive so that manifest `npz_path` entries
   resolve, or rewrite the manifest paths to the restored cache root.
4. Restore V1a.2-A and V1a.2-B run directories from Drive.
5. Clone the MedSAM repository and use the Drive checkpoint.
6. Run only the three follow-up tests listed in
   `V1A2_VALIDATION_STATUS.md`; do not repeat training or the completed
   presence-Pareto runs.

The frozen test split must remain untouched.
