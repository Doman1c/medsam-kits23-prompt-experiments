#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
V1A5_MERGE="af2b92b96c0fecaecbfa21038b9d3aae3de27378"

[[ -d "$REPO/.git" ]] || { echo "ERROR: repo missing: $REPO" >&2; exit 1; }

git -C "$REPO" merge-base --is-ancestor "$V1A5_MERGE" HEAD || {
  echo "ERROR: current branch does not descend from frozen V1a.5 merge" >&2
  exit 1
}

bash "$REPO/v1/v1a5/scripts/restore_v1a5_colab.sh"
source "$WORK/v1a5_env.sh"

python -m py_compile \
  "$REPO/v1/v1a6/v1a6/geometry_consensus.py" \
  "$REPO/v1/v1a6/scripts/materialize_v1a6_candidates.py" \
  "$REPO/v1/v1a6/scripts/test_v1a6_geometry_consensus.py"
python "$REPO/v1/v1a6/scripts/test_v1a6_geometry_consensus.py"

EXPECTED_CKPT_SHA256="f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2"
ACTUAL_CKPT_SHA256="$(sha256sum "$V1A3A_CKPT" | awk '{print $1}')"
[[ "$ACTUAL_CKPT_SHA256" == "$EXPECTED_CKPT_SHA256" ]] || {
  echo "ERROR: frozen checkpoint SHA mismatch" >&2
  exit 1
}

cat > "$WORK/v1a6_env.sh" <<EOF
export REPO="$REPO"
export WORK="$WORK"
export MANIFEST="$MANIFEST"
export V1A3A_CKPT="$V1A3A_CKPT"
export MEDSAM_REPO="$MEDSAM_REPO"
export MEDSAM_CKPT="$MEDSAM_CKPT"
export V1A6_BASE_MERGE="$V1A5_MERGE"
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
    'v1a6_environment_gate': 'PASSED',
    'validation_cases': 20,
    'validation_rows': 3644,
    'checkpoint_sha256': '$ACTUAL_CKPT_SHA256',
    'test_access': False,
})
PY

echo
echo "V1a.6 environment restore complete."
echo "Environment file: $WORK/v1a6_env.sh"
