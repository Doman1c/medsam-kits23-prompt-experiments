#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
ENV_FILE="$WORK/v1a4_env.sh"
DRIVE="${DRIVE:-/content/drive/MyDrive}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Run restore_v1a4_colab.sh first." >&2
  exit 1
fi
source "$ENV_FILE"

OUT="${OUT:-$WORK/v1a4_stage1}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE/medsam-kits23/v1a4_results/stage1}"
EXPECTED_CKPT_SHA256="f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2"

PREDICT="$REPO/v1/v1a2b/scripts/predict_v1a2b.py"
AUDIT="$REPO/v1/v1a4/scripts/audit_recovery_errors.py"
SWEEP="$REPO/v1/v1a4/scripts/sweep_recovery_decoder.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
LOCK="$REPO/v1/v1a4/BASELINE_LOCK.json"

for path in "$MANIFEST" "$V1A3A_CKPT" "$PREDICT" "$AUDIT" "$SWEEP" "$PROMPT_EVAL" "$LOCK"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

ACTUAL_CKPT_SHA256="$(sha256sum "$V1A3A_CKPT" | awk '{print $1}')"
if [[ "$ACTUAL_CKPT_SHA256" != "$EXPECTED_CKPT_SHA256" ]]; then
  echo "ERROR: frozen epoch-18 checkpoint SHA-256 mismatch" >&2
  exit 1
fi

python - <<PY
import json
from pathlib import Path
import pandas as pd
lock = json.loads(Path('$LOCK').read_text(encoding='utf-8'))
assert lock['base']['v1a3b_merge_commit'] == '36c260096f733fc54e4db0819cc04527d021eff6'
assert lock['base']['decoder']['mode'] == 'single_span'
assert abs(float(lock['base']['decoder']['left_threshold']) - 0.0035) < 1e-12
assert abs(float(lock['base']['decoder']['right_threshold']) - 0.04) < 1e-12
assert int(lock['base']['decoder']['baseline_max_gap']) == 4
assert int(lock['base']['decoder']['boundary_pad']) == 1
m = pd.read_csv('$MANIFEST', dtype={'case_id': str})
val = m[m['split'].astype(str) == 'val']
assert val['case_id'].nunique() == 20
assert len(val) == 3644
print({
    'v1a4_stage1_lock': 'PASSED',
    'checkpoint_sha256': '$ACTUAL_CKPT_SHA256',
    'validation_cases': 20,
    'validation_rows': 3644,
    'test_access': False,
})
PY

python -m py_compile \
  "$REPO/v1/v1a4/v1a4/recovery_decoder.py" \
  "$AUDIT" \
  "$SWEEP"
python "$REPO/v1/v1a4/scripts/test_v1a4_recovery_decoder.py"

mkdir -p "$OUT" "$DRIVE_OUT"

resolve_raw() {
  for path in \
    "$WORK/v1a3b_validation/val_epoch18_base_predictions.csv" \
    "$DRIVE/medsam-kits23/v1a3b_results/validation/val_epoch18_base_predictions.csv" \
    "$OUT/val_epoch18_base_predictions.csv"
  do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

RAW="$(resolve_raw || true)"
if [[ -z "$RAW" ]]; then
  RAW="$OUT/val_epoch18_base_predictions.csv"
  python "$PREDICT" \
    --manifest "$MANIFEST" \
    --checkpoint "$V1A3A_CKPT" \
    --output-csv "$RAW" \
    --split val \
    --weights ema \
    --batch-size 32 \
    --num-workers 2 \
    --presence-threshold 0.5 \
    --max-gap 0 \
    --device cuda
fi

echo "Using validation base predictions: $RAW"

rm -rf "$OUT/recovery_audit"
python "$AUDIT" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/recovery_audit"

rm -rf "$OUT/recovery_sweep"
python "$SWEEP" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/recovery_sweep" \
  --left-recovery-thresholds 0.0005,0.001,0.0015,0.002,0.0025,0.003 \
  --right-recovery-thresholds 0.0025,0.005,0.01,0.015,0.02,0.03 \
  --max-extensions 2,4,6,8,12,16 \
  --bridge-gaps 0,1,2 \
  --satellite-max-distances 2,4,6,8,12,16 \
  --satellite-evidence-ratios 0.05,0.1,0.2,0.35,0.5 \
  --satellite-min-lengths 1,2,3 \
  --min-overall-recall 0.96 \
  --min-left-recall 0.94 \
  --min-right-recall 0.97 \
  --max-fpr 0.15 \
  --shortlist-per-mode 3 \
  --expansion-scale 1.20

mapfile -t CANDIDATE_DIRS < <(
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path('$OUT/recovery_sweep/candidate_manifest.json').read_text(encoding='utf-8'))
for item in payload['candidates']:
    print(item['directory'])
PY
)

for candidate_dir in "${CANDIDATE_DIRS[@]}"; do
  rm -rf "$candidate_dir/prompt_metrics"
  python "$PROMPT_EVAL" \
    --predictions "$candidate_dir/recommended_predictions.csv" \
    --output-dir "$candidate_dir/prompt_metrics" \
    --variant smooth \
    --expansion-scale 1.20 \
    --target-labels 1
done

python - <<PY
import json
from pathlib import Path
import pandas as pd
root = Path('$OUT/recovery_sweep')
payload = json.loads((root / 'candidate_manifest.json').read_text(encoding='utf-8'))
rows = []
for item in payload['candidates']:
    candidate_dir = Path(item['directory'])
    prompt = json.loads((candidate_dir / 'prompt_metrics/smooth_summary.json').read_text(encoding='utf-8'))
    presence = item['presence_metrics']
    decoder = item['recovery_decoder']
    rows.append({
        'name': item['name'],
        'mode': decoder['mode'],
        'positive_recall': presence['positive_recall'],
        'left_positive_recall': presence['left_positive_recall'],
        'right_positive_recall': presence['right_positive_recall'],
        'false_positive_rate': presence['false_positive_rate'],
        'specificity': presence['specificity'],
        'multi_segment_case_side_rate': presence['multi_segment_case_side_rate'],
        'prompt_success_rate': prompt['prompt_success_rate'],
        'tight_box_iou': prompt['tight_box_iou'],
        'continuous_score': prompt['continuous_score'],
        'left_recovery_threshold': decoder['left_recovery_threshold'],
        'right_recovery_threshold': decoder['right_recovery_threshold'],
        'max_extension': decoder['max_extension'],
        'bridge_gap': decoder['bridge_gap'],
        'satellite_max_distance': decoder['satellite_max_distance'],
        'satellite_evidence_ratio': decoder['satellite_evidence_ratio'],
        'satellite_min_length': decoder['satellite_min_length'],
    })
frame = pd.DataFrame(rows).sort_values(
    ['positive_recall', 'false_positive_rate', 'prompt_success_rate'],
    ascending=[False, True, False],
)
frame.to_csv(root / 'shortlist_prompt_comparison.csv', index=False)
print(frame.to_string(index=False))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.4 Stage 1 complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "No MedSAM and no test split were used in this stage."
