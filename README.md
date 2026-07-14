# MedSAM Automatic Prompt V0

This repository stores code and small reproducibility artifacts for the
KiTS23 + MedSAM prompt-ablation experiments.

## Repository contents

```text
.
├── prepare_kits23_2d.py
├── run_v0_experiment.py
├── environment_check.py
├── requirements.txt
├── .gitignore
├── selected_cases/
│   └── kits23_selected_100_cases.json
└── results/
    ├── random_100/
    └── one_per_case/
```

Raw KiTS23 volumes, generated NPZ slices, download caches, and
`medsam_vit_b.pth` are intentionally excluded from Git.

## Expected Colab paths

```text
/content/MedSAM
/content/MedSAM/work_dir/MedSAM/medsam_vit_b.pth
/content/kits23/dataset
/content/kits23_2d_all
/content/language_prompt_medsam_starter
/content/v0_kidney_random_100
/content/v0_kidney_one_per_case
```

## Environment check

```bash
python environment_check.py
```

Full processed-file check and checkpoint MD5:

```bash
python environment_check.py \
  --full-npz-scan \
  --md5
```

## Preprocessing

```bash
python prepare_kits23_2d.py \
  --kits-root /content/kits23/dataset \
  --output-root /content/kits23_2d_all \
  --target-ids 1 2 3 \
  --min-target-pixels 20
```

## V0 baseline

```bash
python run_v0_experiment.py \
  --processed-root /content/kits23_2d_all \
  --checkpoint /content/MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
  --output-dir /content/v0_kidney_one_per_case \
  --target-name kidney \
  --sampling one_per_case \
  --component-mode largest \
  --modes box box_mask
```
