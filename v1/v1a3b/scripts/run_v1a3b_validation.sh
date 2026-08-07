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

OUT="${OUT:-$WORK/v1a3b_validation}"
DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/medsam-kits23/v1a3b_results/validation}"
RUN_MEDSAM="${RUN_MEDSAM:-0}"

PREDICT="$REPO/v1/v1a2b/scripts/predict_v1a2b.py"
AUDIT="$REPO/v1/v1a3b/scripts/audit_presence_topology.py"
SWEEP="$REPO/v1/v1a3b/scripts/sweep_span_decoder.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
NEGATIVE_AWARE="$REPO/v1/v1a2b/followup_validation/tools/negative_aware_medsam_compare.py"

for path in "$MANIFEST" "$V1A3A_CKPT" "$PREDICT" "$AUDIT" "$SWEEP" "$PROMPT_EVAL"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

mkdir -p "$OUT" "$DRIVE_OUT"
RAW="$OUT/val_epoch18_base_predictions.csv"

if [[ ! -f "$RAW" ]]; then
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

rm -rf "$OUT/topology_audit"
python "$AUDIT" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/topology_audit" \
  --left-threshold 0.005 \
  --right-threshold 0.005 \
  --max-gap 2

rm -rf "$OUT/span_sweep"
python "$SWEEP" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/span_sweep" \
  --thresholds 0.0025,0.005,0.0075,0.01,0.02,0.04,0.08,0.10,0.15 \
  --baseline-max-gaps 0,1,2 \
  --single-min-span-lengths 1,3 \
  --optimal-max-internal-gaps 1,2,3 \
  --optimal-min-span-lengths 1,3 \
  --optimal-span-penalties 0,1 \
  --optimal-length-penalties 0 \
  --min-overall-recall 0.95 \
  --min-side-recall 0.90 \
  --shortlist-per-mode 2 \
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
    prompt = json.loads((candidate_dir / 'prompt_metrics/smooth_summary.json').read_text(encoding='utf-8'))
    presence = item['presence_metrics']
    rows.append({
        'name': item['name'],
        'mode': item['decoder']['mode'],
        'left_threshold': item['decoder']['left_threshold'],
        'right_threshold': item['decoder']['right_threshold'],
        'positive_recall': presence['positive_recall'],
        'left_positive_recall': presence['left_positive_recall'],
        'right_positive_recall': presence['right_positive_recall'],
        'false_positive_rate': presence['false_positive_rate'],
        'specificity': presence['specificity'],
        'pred_multi_segment_case_side_rate': presence['multi_segment_case_side_rate'],
        'prompt_success_rate': prompt['prompt_success_rate'],
        'tight_box_iou': prompt['tight_box_iou'],
        'continuous_score': prompt['continuous_score'],
    })
frame = pd.DataFrame(rows).sort_values(
    ['positive_recall', 'false_positive_rate', 'prompt_success_rate'],
    ascending=[False, True, False],
)
frame.to_csv(root / 'shortlist_prompt_comparison.csv', index=False)
print(frame.to_string(index=False))
PY

if [[ "$RUN_MEDSAM" == "1" ]]; then
  [[ -f "$MEDSAM_CKPT" ]] || { echo "ERROR: missing MedSAM checkpoint" >&2; exit 1; }
  [[ -d "$MEDSAM_REPO/segment_anything" ]] || { echo "ERROR: invalid MedSAM repository" >&2; exit 1; }
  rm -rf "$OUT/negative_aware_medsam"
  python "$NEGATIVE_AWARE" \
    --repo-root "$REPO" \
    --candidate-manifest "$OUT/span_sweep/candidate_manifest.json" \
    --medsam-repo "$MEDSAM_REPO" \
    --checkpoint "$MEDSAM_CKPT" \
    --output-dir "$OUT/negative_aware_medsam" \
    --target-labels 1 \
    --device cuda
fi

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.3-B validation stage complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "RUN_MEDSAM=$RUN_MEDSAM"
