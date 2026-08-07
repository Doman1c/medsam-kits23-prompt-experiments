#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
ENV_FILE="$WORK/v1a6_env.sh"
DRIVE="${DRIVE:-/content/drive/MyDrive}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Run restore_v1a6_colab.sh first." >&2
  exit 1
fi
source "$ENV_FILE"

STAGE1="${STAGE1:-$WORK/v1a6_stage1}"
OUT="${OUT:-$WORK/v1a6_stage2_medsam}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE/medsam-kits23/v1a6_results/stage2_medsam}"
EXPECTED_CKPT_SHA256="f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2"

PREPARE="$REPO/v1/v1a6/scripts/prepare_v1a6_stage2_candidates.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
NEGATIVE_AWARE="$REPO/v1/v1a2b/followup_validation/tools/negative_aware_medsam_compare.py"
BASE_LOCK="$REPO/v1/v1a6/BASELINE_LOCK.json"
STAGE2_LOCK="$REPO/v1/v1a6/STAGE2_LOCK.json"
RECOVERED_KEYS="$STAGE1/candidate_set/recovered_side_keys.csv"

for path in \
  "$MANIFEST" "$V1A3A_CKPT" "$MEDSAM_CKPT" "$PREPARE" "$PROMPT_EVAL" \
  "$NEGATIVE_AWARE" "$BASE_LOCK" "$STAGE2_LOCK" \
  "$STAGE1/candidate_set/candidate_manifest.json" "$STAGE1/stage1_geometry_gate.csv" \
  "$RECOVERED_KEYS"
do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

[[ -d "$MEDSAM_REPO/segment_anything" ]] || {
  echo "ERROR: invalid MedSAM repository: $MEDSAM_REPO" >&2
  exit 1
}

ACTUAL_CKPT_SHA256="$(sha256sum "$V1A3A_CKPT" | awk '{print $1}')"
[[ "$ACTUAL_CKPT_SHA256" == "$EXPECTED_CKPT_SHA256" ]] || {
  echo "ERROR: frozen epoch-18 checkpoint SHA-256 mismatch" >&2
  exit 1
}

resolve_b1_dir() {
  for path in \
    "$WORK/v1a4_stage2_medsam/candidate_set/candidates/v1a3b_frozen_b1" \
    "$DRIVE/medsam-kits23/v1a4_results/stage2_medsam/candidate_set/candidates/v1a3b_frozen_b1" \
    "$WORK/v1a4_stage1b/recovery_sweep/candidates/v1a3b_frozen_b1" \
    "$DRIVE/medsam-kits23/v1a4_results/stage1b/recovery_sweep/candidates/v1a3b_frozen_b1"
  do
    if [[ -f "$path/recommended_predictions.csv" && -f "$path/recommended_calibration.json" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

B1_DIR="$(resolve_b1_dir || true)"
if [[ -z "$B1_DIR" ]]; then
  echo "ERROR: frozen V1a.3-B B1 validation candidate directory not found." >&2
  exit 1
fi

echo "Using frozen B1: $B1_DIR"

python - <<PY
from pathlib import Path
import json
import pandas as pd
base = json.loads(Path('$BASE_LOCK').read_text(encoding='utf-8'))
lock = json.loads(Path('$STAGE2_LOCK').read_text(encoding='utf-8'))
assert base['base']['v1a5_merge_commit'] == 'af2b92b96c0fecaecbfa21038b9d3aae3de27378'
assert lock['status'] == 'stage2_validation_candidates_frozen_before_medsam'
assert lock['test_set_used'] is False
assert lock['candidate_selection']['performed_before_medsam'] is True
assert [x['name'] for x in lock['candidate_selection']['candidates']] == [
    'g0_frozen_b1', 'g1_r2b_native', 'g6_robust_linear'
]
gate = pd.read_csv('$STAGE1/stage1_geometry_gate.csv')
g6 = gate[gate['name'] == 'g6_robust_linear']
assert len(g6) == 1 and bool(g6.iloc[0]['advance_to_medsam_review'])
m = pd.read_csv('$MANIFEST', dtype={'case_id': str})
val = m[m['split'].astype(str) == 'val']
assert val['case_id'].nunique() == 20 and len(val) == 3644
print({
    'v1a6_stage2_gate': 'PASSED',
    'validation_cases': 20,
    'validation_rows': 3644,
    'checkpoint_sha256': '$ACTUAL_CKPT_SHA256',
    'candidate_count': 3,
    'test_access': False,
})
PY

rm -rf "$OUT"
mkdir -p "$OUT" "$DRIVE_OUT"

python "$PREPARE" \
  --stage1-root "$STAGE1" \
  --b1-dir "$B1_DIR" \
  --output-dir "$OUT/candidate_set" \
  --expansion-scale 1.20

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

python "$NEGATIVE_AWARE" \
  --repo-root "$REPO" \
  --candidate-manifest "$OUT/candidate_set/candidate_manifest.json" \
  --medsam-repo "$MEDSAM_REPO" \
  --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$OUT/negative_aware_medsam" \
  --target-labels 1 \
  --device cuda

python - <<PY
from pathlib import Path
import json
import pandas as pd

out = Path('$OUT')
manifest = json.loads((out / 'candidate_set/candidate_manifest.json').read_text(encoding='utf-8'))
medsam = pd.read_csv(out / 'negative_aware_medsam/negative_aware_candidate_comparison.csv')
rows = []
for item in manifest['candidates']:
    summary = json.loads((Path(item['directory']) / 'prompt_metrics/smooth_summary.json').read_text(encoding='utf-8'))
    rows.append({
        'name': item['name'],
        'presence_recall': summary['positive_recall'],
        'presence_fpr': summary['false_positive_rate'],
        'prompt_success_rate': summary['prompt_success_rate'],
        'tight_box_iou': summary['tight_box_iou'],
        'continuous_score': summary['continuous_score'],
    })
prompt = pd.DataFrame(rows)
combined = prompt.merge(medsam, left_on='name', right_on='candidate', how='left', validate='one_to_one')
combined = combined.sort_values(
    ['case_volume_dice_mean', 'all_slice_dice_mean', 'negative_false_positive_mask_rate'],
    ascending=[False, False, True],
).reset_index(drop=True)
combined.to_csv(out / 'stage2_combined_comparison.csv', index=False)
print('\nCombined Stage-2 comparison:')
print(combined.to_string(index=False))

# Paired analysis on exactly the R2b-recovered validation slices. The Stage-1
# recovered side set is left-only, so its unique (case,slice) keys map directly
# to the combined MedSAM slice table.
recovered = pd.read_csv('$RECOVERED_KEYS', dtype={'case_id': str})
recovered = recovered[recovered['gt_present'].astype(int) == 1][['case_id', 'slice_idx']].drop_duplicates()
details = pd.read_csv(out / 'negative_aware_medsam/negative_aware_medsam_details.csv', dtype={'case_id': str})
subset = details.merge(recovered, on=['case_id', 'slice_idx'], how='inner')
pivot = subset.pivot(index=['case_id','slice_idx'], columns='candidate', values=['dice','tp_pixels','fp_pixels','fn_pixels']).reset_index()
for metric in ('dice','tp_pixels','fp_pixels','fn_pixels'):
    pivot[(metric, 'g6_minus_g1')] = pivot[(metric, 'g6_robust_linear')] - pivot[(metric, 'g1_r2b_native')]
pivot.columns = ['__'.join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in pivot.columns]
pivot.to_csv(out / 'stage2_recovered_slice_paired.csv', index=False)

paired_summary = {
    'recovered_unique_slices': int(len(recovered)),
    'g1_recovered_dice_mean': float(subset[subset['candidate']=='g1_r2b_native']['dice'].mean()),
    'g6_recovered_dice_mean': float(subset[subset['candidate']=='g6_robust_linear']['dice'].mean()),
    'g1_recovered_dice_median': float(subset[subset['candidate']=='g1_r2b_native']['dice'].median()),
    'g6_recovered_dice_median': float(subset[subset['candidate']=='g6_robust_linear']['dice'].median()),
    'g1_recovered_tp_pixels': int(subset[subset['candidate']=='g1_r2b_native']['tp_pixels'].sum()),
    'g6_recovered_tp_pixels': int(subset[subset['candidate']=='g6_robust_linear']['tp_pixels'].sum()),
    'g1_recovered_fp_pixels': int(subset[subset['candidate']=='g1_r2b_native']['fp_pixels'].sum()),
    'g6_recovered_fp_pixels': int(subset[subset['candidate']=='g6_robust_linear']['fp_pixels'].sum()),
}
(out / 'stage2_recovered_slice_summary.json').write_text(json.dumps(paired_summary, indent=2), encoding='utf-8')
print('\nRecovered-slice paired summary:')
print(json.dumps(paired_summary, indent=2))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.6 Stage 2 negative-aware MedSAM validation complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "No test or secondary-test split was used."
