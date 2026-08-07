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

OUT="${OUT:-$WORK/v1a4_stage1b}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE/medsam-kits23/v1a4_results/stage1b}"
EXPECTED_CKPT_SHA256="f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2"

SWEEP="$REPO/v1/v1a4/scripts/sweep_recovery_stage1b.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
TEST_DECODER="$REPO/v1/v1a4/scripts/test_v1a4_stage1b_decoder.py"
STAGE1B_DECODER="$REPO/v1/v1a4/v1a4/stage1b_decoder.py"
LOCK="$REPO/v1/v1a4/BASELINE_LOCK.json"

for path in "$MANIFEST" "$V1A3A_CKPT" "$SWEEP" "$PROMPT_EVAL" "$TEST_DECODER" "$STAGE1B_DECODER" "$LOCK"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

ACTUAL_CKPT_SHA256="$(sha256sum "$V1A3A_CKPT" | awk '{print $1}')"
if [[ "$ACTUAL_CKPT_SHA256" != "$EXPECTED_CKPT_SHA256" ]]; then
  echo "ERROR: frozen epoch-18 checkpoint SHA-256 mismatch" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
import json
import pandas as pd
lock = json.loads(Path('$LOCK').read_text(encoding='utf-8'))
assert lock['base']['v1a3b_merge_commit'] == '36c260096f733fc54e4db0819cc04527d021eff6'
m = pd.read_csv('$MANIFEST', dtype={'case_id': str})
val = m[m['split'].astype(str) == 'val']
assert val['case_id'].nunique() == 20
assert len(val) == 3644
missing = [p for p in val['npz_path'].astype(str) if not Path(p).is_file()]
assert not missing, f'{len(missing)} validation NPZ files missing; examples={missing[:5]}'
print({
    'v1a4_stage1b_lock': 'PASSED',
    'checkpoint_sha256': '$ACTUAL_CKPT_SHA256',
    'validation_cases': 20,
    'validation_rows': 3644,
    'test_access': False,
})
PY

python -m py_compile "$STAGE1B_DECODER" "$SWEEP" "$TEST_DECODER"
python "$TEST_DECODER"

mkdir -p "$OUT" "$DRIVE_OUT"

resolve_raw() {
  for path in \
    "$WORK/v1a3b_validation/val_epoch18_base_predictions.csv" \
    "$WORK/v1a4_stage1/val_epoch18_base_predictions.csv" \
    "$DRIVE/medsam-kits23/v1a3b_results/validation/val_epoch18_base_predictions.csv" \
    "$DRIVE/medsam-kits23/v1a4_results/stage1/val_epoch18_base_predictions.csv"
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
  echo "ERROR: validation base predictions not found. Run V1a.4 Stage 1 first." >&2
  exit 1
fi

echo "Using validation base predictions: $RAW"

rm -rf "$OUT/recovery_sweep"
python "$SWEEP" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/recovery_sweep" \
  --satellite-max-distances 20,24,32,48,64,96 \
  --satellite-evidence-ratios 0.05,0.1,0.2,0.35,0.5,0.75 \
  --satellite-min-lengths 1,2,3,5,8 \
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
    decoder = item['decoder']
    rows.append({
        'name': item['name'],
        'family': item['family'],
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
        'satellite_max_distance': decoder.get('satellite_max_distance', 0),
        'satellite_evidence_ratio': decoder.get('satellite_evidence_ratio', 0.0),
        'satellite_min_length': decoder.get('satellite_min_length', 1),
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
echo "V1a.4 Stage 1b complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "No MedSAM and no test split were used in this stage."
