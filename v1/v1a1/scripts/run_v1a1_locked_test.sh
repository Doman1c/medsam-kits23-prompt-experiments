#!/usr/bin/env bash
set -euo pipefail

# Frozen V1a.1 test evaluation. This script performs no threshold/checkpoint/scale sweep.
REPO_ROOT="/content/medsam-kits23-prompt-experiments"
PROJECT="$REPO_ROOT/v1/v1a1"
WORK="/content/v1a1_work"
MANIFEST="$WORK/manifest/v1a1_dual_manifest.csv"
CHECKPOINT="/content/drive/MyDrive/medsam-kits23/v1a1_results/run/best_continuous.pt"
CAL_JSON="$PROJECT/results/recommended_final_validation_config.json"
MEDSAM_REPO="$WORK/MedSAM"
MEDSAM_CKPT="$WORK/medsam_weights/medsam_vit_b.pth"
OUT="$WORK/test_final"
DRIVE_OUT="/content/drive/MyDrive/medsam-kits23/v1a1_results/test_final"

EXPECTED_COMMIT="be3e114215aac33525e846d004b66088e62e92d5"

cd "$REPO_ROOT"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if ! git merge-base --is-ancestor "$EXPECTED_COMMIT" "$ACTUAL_COMMIT"; then
  echo "ERROR: frozen base commit $EXPECTED_COMMIT is not an ancestor of $ACTUAL_COMMIT" >&2
  exit 1
fi
cd "$PROJECT"

for path in "$MANIFEST" "$CHECKPOINT" "$CAL_JSON"; do
  [[ -f "$path" ]] || { echo "ERROR: missing file: $path" >&2; exit 1; }
done

mkdir -p "$OUT"

# Pre-test lock audit. This runs before any test prediction is exported.
python - <<'PY'
import hashlib, json
from pathlib import Path
import pandas as pd
import torch

manifest_path = Path('/content/v1a1_work/manifest/v1a1_dual_manifest.csv')
checkpoint_path = Path('/content/drive/MyDrive/medsam-kits23/v1a1_results/run/best_continuous.pt')
calibration_path = Path('/content/medsam-kits23-prompt-experiments/v1/v1a1/results/recommended_final_validation_config.json')
out_path = Path('/content/v1a1_work/test_final/test_protocol_lock.json')

try:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
except TypeError:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
cal = json.loads(calibration_path.read_text(encoding='utf-8'))
manifest = pd.read_csv(manifest_path)
test = manifest[manifest['split'].astype(str) == 'test'].copy()

assert int(ckpt['epoch']) == 6, ckpt.get('epoch')
assert 'ema_state' in ckpt
assert ckpt.get('config', {}).get('input_mode', '2.5d') == '2.5d'
assert cal['candidate'] == 'epoch6'
assert cal['variant'] == 'smooth'
assert int(cal['max_gap']) == 0
assert abs(float(cal['left_threshold']) - 0.20) < 1e-12
assert abs(float(cal['right_threshold']) - 0.30) < 1e-12
assert abs(float(cal['expansion_scale']) - 1.20) < 1e-12
assert test['case_id'].astype(str).nunique() == 27
assert len(test) == 4277
missing_npz = [path for path in test['npz_path'].astype(str) if not Path(path).is_file()]
assert not missing_npz, (
    f"{len(missing_npz)} test NPZ files are missing; "
    f"first examples: {missing_npz[:5]}"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

lock = {
    'status': 'locked_before_test',
    'checkpoint_epoch': int(ckpt['epoch']),
    'weights': 'ema',
    'configuration': {
        'variant': 'smooth',
        'max_gap': 0,
        'left_threshold': 0.20,
        'right_threshold': 0.30,
        'expansion_scale': 1.20,
        'target_labels': [1],
    },
    'expected_test': {'cases': 27, 'rows': 4277, 'positive_slices': 1473},
    'sha256': {
        'manifest': sha256(manifest_path),
        'checkpoint': sha256(checkpoint_path),
        'calibration_json': sha256(calibration_path),
    },
}
out_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(lock, indent=2, ensure_ascii=False))
PY

# 1) Export base test predictions. The 0.5 CLI threshold is only a placeholder for
# raw_presence columns; the formal smooth predictions are recalculated in step 2
# with the frozen independent left/right thresholds.
python scripts/predict_dual.py \
  --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output-csv "$OUT/test_base_predictions.csv" \
  --split test \
  --weights ema \
  --batch-size 32 \
  --num-workers 2 \
  --presence-threshold 0.5 \
  --max-gap 0 \
  --device cuda

# 2) Apply the frozen validation calibration: smooth, max_gap=0, left=0.20, right=0.30.
python scripts/apply_saved_calibration.py \
  --predictions "$OUT/test_base_predictions.csv" \
  --calibration-json "$CAL_JSON" \
  --output-csv "$OUT/test_calibrated_predictions.csv"

# 3) Prompt-level test metrics. Only smooth is the formal automatic configuration.
python scripts/evaluate_dual_prompts.py \
  --predictions "$OUT/test_calibrated_predictions.csv" \
  --output-dir "$OUT/prompt_metrics" \
  --variant smooth \
  --expansion-scale 1.20 \
  --target-labels 1

# 4) Prepare frozen MedSAM assets if not already restored.
if [[ ! -d "$MEDSAM_REPO/segment_anything" || ! -f "$MEDSAM_CKPT" ]]; then
  python scripts/setup_medsam.py \
    --repo-dir "$MEDSAM_REPO" \
    --checkpoint-source /content/drive/MyDrive/medsam_vit_b.pth \
    --checkpoint-dest "$MEDSAM_CKPT"
fi

# 5) Full positive-slice MedSAM test evaluation.
# Deliberately omit --max-samples and --include-negatives.
python scripts/evaluate_medsam_dual.py \
  --predictions "$OUT/test_calibrated_predictions.csv" \
  --medsam-repo "$MEDSAM_REPO" \
  --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$OUT/medsam" \
  --target-labels 1 \
  --expansion-scale 1.20 \
  --device cuda

# 6) Produce the final locked report and error-analysis tables.
python scripts/summarize_locked_test.py \
  --manifest "$MANIFEST" \
  --predictions "$OUT/test_calibrated_predictions.csv" \
  --calibration-json "$CAL_JSON" \
  --checkpoint "$CHECKPOINT" \
  --medsam-details "$OUT/medsam/medsam_dual_details.csv" \
  --output-dir "$OUT/report" \
  --expected-test-cases 27 \
  --expected-test-rows 4277 \
  --expected-positive-slices 1473 \
  --figure-count 4

# 7) Back up the complete immutable result bundle to Drive.
mkdir -p "$DRIVE_OUT"
rsync -a --delete "$OUT/" "$DRIVE_OUT/"

echo "Completed frozen V1a.1 test evaluation."
echo "Local: $OUT"
echo "Drive: $DRIVE_OUT"
