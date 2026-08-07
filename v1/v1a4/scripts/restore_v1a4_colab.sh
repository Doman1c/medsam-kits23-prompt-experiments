#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
V1A3B_MERGE="36c260096f733fc54e4db0819cc04527d021eff6"

if [[ ! -d "$REPO/.git" ]]; then
  echo "ERROR: repository not found at $REPO" >&2
  exit 1
fi

cd "$REPO"
if ! git merge-base --is-ancestor "$V1A3B_MERGE" HEAD; then
  echo "ERROR: repository HEAD does not contain frozen V1a.3-B merge $V1A3B_MERGE" >&2
  exit 1
fi

# Reuse the already-tested V1a.3-B asset restore: manifest, NPZ paths,
# V1a.3-A epoch-18 checkpoint, MedSAM repo, and MedSAM checkpoint.
bash "$REPO/v1/v1a3b/scripts/restore_v1a3b_colab.sh"
source "$WORK/v1a3b_env.sh"

python "$REPO/v1/v1a4/scripts/test_v1a4_recovery_decoder.py"

ENV_FILE="$WORK/v1a4_env.sh"
cat > "$ENV_FILE" <<EOF
export REPO="$REPO"
export WORK="$WORK"
export MANIFEST="$MANIFEST"
export V1A3A_CKPT="$V1A3A_CKPT"
export MEDSAM_REPO="$MEDSAM_REPO"
export MEDSAM_CKPT="$MEDSAM_CKPT"
export V1A4_BASE_MERGE="$V1A3B_MERGE"
EOF

python - <<PY
from pathlib import Path
import pandas as pd
m = pd.read_csv('$MANIFEST', dtype={'case_id': str})
val = m[m['split'].astype(str) == 'val']
assert val['case_id'].nunique() == 20
assert len(val) == 3644
missing = [p for p in val['npz_path'].astype(str) if not Path(p).is_file()]
assert not missing, f'{len(missing)} validation NPZ files missing; examples={missing[:5]}'
print({
    'v1a4_environment_gate': 'PASSED',
    'validation_cases': 20,
    'validation_rows': 3644,
    'test_access': False,
})
PY

cat "$ENV_FILE"
echo
echo "V1a.4 environment restore complete."
echo "Source it with: source $ENV_FILE"
