#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
ENV_FILE="$WORK/v1a5_env.sh"
DRIVE="${DRIVE:-/content/drive/MyDrive}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Run restore_v1a5_colab.sh first." >&2
  exit 1
fi
source "$ENV_FILE"

OUT="${OUT:-$WORK/v1a5_stage1}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE/medsam-kits23/v1a5_results/stage1}"
MATERIALIZE="$REPO/v1/v1a5/scripts/materialize_v1a5_geometry_candidates.py"
AUDIT="$REPO/v1/v1a5/scripts/audit_v1a5_geometry.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"

resolve_candidate() {
  local name="$1"
  for path in \
    "$WORK/v1a4_stage2_medsam/candidate_set/candidates/$name/recommended_predictions.csv" \
    "$DRIVE/medsam-kits23/v1a4_results/stage2_medsam/candidate_set/candidates/$name/recommended_predictions.csv" \
    "$WORK/v1a4_stage1b/recovery_sweep/candidates/$name/recommended_predictions.csv" \
    "$DRIVE/medsam-kits23/v1a4_results/stage1b/recovery_sweep/candidates/$name/recommended_predictions.csv"
  do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

B1="$(resolve_candidate v1a3b_frozen_b1 || true)"
R2B="$(resolve_candidate r2b_remote_satellite_1 || true)"

if [[ -z "$B1" || -z "$R2B" ]]; then
  echo "ERROR: frozen B1/R2b validation predictions not found." >&2
  echo "Run V1a.4 Stage 1b/Stage 2 once in this runtime or ensure the Drive backups are mounted." >&2
  exit 1
fi

echo "Using B1:  $B1"
echo "Using R2b: $R2B"

rm -rf "$OUT"
mkdir -p "$OUT" "$DRIVE_OUT"

python "$MATERIALIZE" \
  --b1-predictions "$B1" \
  --r2b-predictions "$R2B" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/candidate_set" \
  --trend-anchor-count 5

python "$AUDIT" \
  --candidate-manifest "$OUT/candidate_set/candidate_manifest.json" \
  --recovered-keys "$OUT/candidate_set/recovered_side_keys.csv" \
  --output-dir "$OUT/geometry_audit"

mapfile -t CANDIDATE_DIRS < <(
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path('$OUT/candidate_set/candidate_manifest.json').read_text(encoding='utf-8'))
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
import numpy as np
import pandas as pd

root = Path('$OUT')
manifest = json.loads((root / 'candidate_set/candidate_manifest.json').read_text(encoding='utf-8'))
recovered = pd.read_csv(root / 'candidate_set/recovered_side_keys.csv', dtype={'case_id': str})
keys = recovered[recovered['gt_present'].astype(int) == 1][['case_id', 'slice_idx', 'side']].copy()
keys['case_id'] = keys['case_id'].astype(str)

rows = []
for item in manifest['candidates']:
    candidate_dir = Path(item['directory'])
    summary = json.loads((candidate_dir / 'prompt_metrics/smooth_summary.json').read_text(encoding='utf-8'))
    details = pd.read_csv(candidate_dir / 'prompt_metrics/smooth_details.csv', dtype={'case_id': str})
    rec = details.merge(keys, on=['case_id', 'slice_idx', 'side'], how='inner', validate='one_to_one')
    finite_center = rec['center_error_fraction'].replace([np.inf, -np.inf], np.nan)
    rows.append({
        'name': item['name'],
        'global_positive_recall': summary['positive_recall'],
        'global_false_positive_rate': summary['false_positive_rate'],
        'global_prompt_success_rate': summary['prompt_success_rate'],
        'global_tight_box_iou': summary['tight_box_iou'],
        'global_continuous_score': summary['continuous_score'],
        'recovered_side_samples': int(len(rec)),
        'recovered_pred_present_rate': float(rec['pred_present'].mean()) if len(rec) else float('nan'),
        'recovered_prompt_success_rate': float(rec['prompt_success'].mean()) if len(rec) else float('nan'),
        'recovered_tight_box_iou_mean': float(rec['tight_box_iou'].mean()) if len(rec) else float('nan'),
        'recovered_tight_box_iou_median': float(rec['tight_box_iou'].median()) if len(rec) else float('nan'),
        'recovered_center_error_fraction_mean': float(finite_center.mean()) if finite_center.notna().any() else float('nan'),
        'recovered_center_error_fraction_median': float(finite_center.median()) if finite_center.notna().any() else float('nan'),
        'recovered_expanded_coverage_mean': float(rec['expanded_coverage'].mean()) if len(rec) else float('nan'),
    })
frame = pd.DataFrame(rows)
frame.to_csv(root / 'stage1_prompt_comparison.csv', index=False)
print(frame.to_string(index=False))

if 'g1_r2b_native' in set(frame['name']):
    baseline = frame[frame['name'] == 'g1_r2b_native'].iloc[0]
    decision_rows = []
    for _, row in frame[frame['name'].isin(['g2_nearest_anchor', 'g3_linear_trend'])].iterrows():
        improves_iou = bool(row['recovered_tight_box_iou_mean'] > baseline['recovered_tight_box_iou_mean'])
        improves_center = bool(
            row['recovered_center_error_fraction_median'] < baseline['recovered_center_error_fraction_median']
        )
        improves_prompt_success = bool(
            row['recovered_prompt_success_rate'] > baseline['recovered_prompt_success_rate']
        )
        decision_rows.append({
            'name': row['name'],
            'improves_recovered_iou': improves_iou,
            'improves_recovered_center': improves_center,
            'improves_recovered_prompt_success': improves_prompt_success,
            'advance_to_medsam_review': bool(improves_iou and improves_center),
        })
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(root / 'stage1_geometry_gate.csv', index=False)
    print('\nGeometry gate:')
    print(decisions.to_string(index=False))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.5 Stage 1 geometry audit complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "No MedSAM and no test split were used."
