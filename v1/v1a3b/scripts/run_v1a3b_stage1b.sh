#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
ENV_FILE="$WORK/v1a3b_env.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Run restore_v1a3b_colab.sh first." >&2
  exit 1
fi
source "$ENV_FILE"

STAGE1_ROOT="${STAGE1_ROOT:-$WORK/v1a3b_validation}"
OUT="${OUT:-$WORK/v1a3b_stage1b}"
DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/medsam-kits23/v1a3b_results/stage1b}"

PREDICT="$REPO/v1/v1a2b/scripts/predict_v1a2b.py"
SWEEP="$REPO/v1/v1a3b/scripts/sweep_span_decoder_stage1b.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"

for path in "$MANIFEST" "$V1A3A_CKPT" "$PREDICT" "$SWEEP" "$PROMPT_EVAL"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

mkdir -p "$OUT" "$DRIVE_OUT"

RAW="$STAGE1_ROOT/val_epoch18_base_predictions.csv"
if [[ ! -f "$RAW" ]]; then
  DRIVE_RAW="/content/drive/MyDrive/medsam-kits23/v1a3b_results/validation/val_epoch18_base_predictions.csv"
  if [[ -f "$DRIVE_RAW" ]]; then
    mkdir -p "$STAGE1_ROOT"
    cp "$DRIVE_RAW" "$RAW"
  else
    mkdir -p "$STAGE1_ROOT"
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
fi

rm -rf "$OUT/span_sweep"
python "$SWEEP" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/span_sweep" \
  --left-thresholds 0.0005,0.001,0.0015,0.002,0.0025,0.0035,0.005 \
  --right-thresholds 0.001,0.0025,0.005,0.0075,0.01,0.02,0.04 \
  --single-max-gaps 2,3,4,5 \
  --optimal-max-internal-gaps 2,3,4,5,6 \
  --boundary-pads 0,1,2,3,4,6,8 \
  --min-overall-recall 0.95 \
  --min-side-recall 0.90 \
  --shortlist-per-mode 3 \
  --expansion-scale 1.20

mapfile -t CANDIDATE_DIRS < <(
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path('$OUT/span_sweep/candidate_manifest.json').read_text(encoding='utf-8'))
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

root = Path('$OUT/span_sweep')
payload = json.loads((root / 'candidate_manifest.json').read_text(encoding='utf-8'))
rows = []
for item in payload['candidates']:
    candidate_dir = Path(item['directory'])
    prompt = json.loads(
        (candidate_dir / 'prompt_metrics/smooth_summary.json').read_text(encoding='utf-8')
    )
    presence = item['presence_metrics']
    decoder = item['decoder']
    rows.append({
        'name': item['name'],
        'mode': decoder['mode'],
        'left_threshold': decoder['left_threshold'],
        'right_threshold': decoder['right_threshold'],
        'baseline_max_gap': decoder.get('baseline_max_gap'),
        'max_internal_gap': decoder.get('max_internal_gap'),
        'boundary_pad': decoder.get('boundary_pad', 0),
        'positive_recall': presence['positive_recall'],
        'left_positive_recall': presence['left_positive_recall'],
        'right_positive_recall': presence['right_positive_recall'],
        'false_positive_rate': presence['false_positive_rate'],
        'specificity': presence['specificity'],
        'onset_abs_error_mean': presence['onset_abs_error_mean'],
        'offset_abs_error_mean': presence['offset_abs_error_mean'],
        'prompt_success_rate': prompt['prompt_success_rate'],
        'tight_box_iou': prompt['tight_box_iou'],
        'continuous_score': prompt['continuous_score'],
    })

frame = pd.DataFrame(rows)
frame['constraints_met'] = (
    (frame['positive_recall'] >= 0.95)
    & (frame['left_positive_recall'] >= 0.90)
    & (frame['right_positive_recall'] >= 0.90)
)
frame = frame.sort_values(
    ['constraints_met', 'false_positive_rate', 'positive_recall', 'prompt_success_rate'],
    ascending=[False, True, False, False],
)
frame.to_csv(root / 'shortlist_prompt_comparison.csv', index=False)
print(frame.to_string(index=False))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.3-B Stage 1b complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
