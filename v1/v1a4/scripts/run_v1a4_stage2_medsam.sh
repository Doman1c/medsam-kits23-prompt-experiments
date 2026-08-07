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

STAGE1B="${STAGE1B:-$WORK/v1a4_stage1b/recovery_sweep}"
OUT="${OUT:-$WORK/v1a4_stage2_medsam}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE/medsam-kits23/v1a4_results/stage2_medsam}"
EXPECTED_CKPT_SHA256="f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2"

PREPARE="$REPO/v1/v1a4/scripts/prepare_v1a4_stage2_candidates.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
NEGATIVE_AWARE="$REPO/v1/v1a2b/followup_validation/tools/negative_aware_medsam_compare.py"
BASE_LOCK="$REPO/v1/v1a4/BASELINE_LOCK.json"
STAGE2_LOCK="$REPO/v1/v1a4/STAGE2_LOCK.json"

for path in \
  "$MANIFEST" \
  "$V1A3A_CKPT" \
  "$STAGE1B/candidate_manifest.json" \
  "$PREPARE" \
  "$PROMPT_EVAL" \
  "$NEGATIVE_AWARE" \
  "$MEDSAM_CKPT" \
  "$BASE_LOCK" \
  "$STAGE2_LOCK"
do
  [[ -e "$path" ]] || {
    echo "ERROR: missing $path" >&2
    exit 1
  }
done

[[ -d "$MEDSAM_REPO/segment_anything" ]] || {
  echo "ERROR: invalid MedSAM repository: $MEDSAM_REPO" >&2
  exit 1
}

ACTUAL_CKPT_SHA256="$(sha256sum "$V1A3A_CKPT" | awk '{print $1}')"
if [[ "$ACTUAL_CKPT_SHA256" != "$EXPECTED_CKPT_SHA256" ]]; then
  echo "ERROR: frozen epoch-18 checkpoint SHA-256 mismatch" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
import json
import pandas as pd
base_lock = json.loads(Path('$BASE_LOCK').read_text(encoding='utf-8'))
stage2_lock = json.loads(Path('$STAGE2_LOCK').read_text(encoding='utf-8'))
assert base_lock['base']['v1a3b_merge_commit'] == '36c260096f733fc54e4db0819cc04527d021eff6'
assert stage2_lock['status'] == 'stage2_validation_candidates_frozen_before_medsam'
assert stage2_lock['test_set_used'] is False
assert stage2_lock['candidate_selection']['performed_before_medsam'] is True
assert [x['name'] for x in stage2_lock['candidate_selection']['candidates']] == [
    'v1a3b_frozen_b1',
    'stage1_fixed_local_hysteresis',
    'r2b_remote_satellite_1',
    'r3_hybrid_1',
]
m = pd.read_csv('$MANIFEST', dtype={'case_id': str})
val = m[m['split'].astype(str) == 'val']
assert val['case_id'].nunique() == 20
assert len(val) == 3644
print({
    'v1a4_stage2_gate': 'PASSED',
    'validation_cases': 20,
    'validation_rows': 3644,
    'checkpoint_sha256': '$ACTUAL_CKPT_SHA256',
    'candidate_count': 4,
    'test_access': False,
})
PY

rm -rf "$OUT"
mkdir -p "$OUT" "$DRIVE_OUT"

python "$PREPARE" \
  --stage1b-root "$STAGE1B" \
  --output-dir "$OUT/candidate_set" \
  --expansion-scale 1.20

mapfile -t CANDIDATE_DIRS < <(
  python - <<PY
import json
from pathlib import Path
payload = json.loads(
    Path('$OUT/candidate_set/candidate_manifest.json').read_text(encoding='utf-8')
)
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
import pandas as pd

out = Path('$OUT')
presence = pd.read_csv(out / 'candidate_set/stage2_presence_candidates.csv')
medsam = pd.read_csv(out / 'negative_aware_medsam/negative_aware_candidate_comparison.csv')
if 'candidate' not in medsam.columns:
    raise RuntimeError(f'Expected candidate column, got {list(medsam.columns)}')
merged = presence.merge(
    medsam,
    left_on='name',
    right_on='candidate',
    how='left',
    validate='one_to_one',
)
merged.to_csv(out / 'stage2_combined_comparison.csv', index=False)
print(merged.to_string(index=False))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.4 Stage 2 negative-aware MedSAM validation complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "No test split was used."
