#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
DRIVE="${DRIVE:-/content/drive/MyDrive}"
BASE_COMMIT="f91422d8b4b44b546a42c649cde92a49d83f07f7"

if [[ ! -d "$DRIVE" ]]; then
  echo "ERROR: Google Drive is not mounted at $DRIVE" >&2
  echo "Run in Colab first:" >&2
  echo "  from google.colab import drive; drive.mount('/content/drive')" >&2
  exit 1
fi

if [[ ! -d "$REPO/.git" ]]; then
  echo "ERROR: repository not found at $REPO" >&2
  exit 1
fi

cd "$REPO"
if ! git merge-base --is-ancestor "$BASE_COMMIT" HEAD; then
  echo "ERROR: repository HEAD does not contain frozen V1a.3-A base $BASE_COMMIT" >&2
  exit 1
fi

python -m pip install -q -r "$REPO/v1/v1a3b/requirements.txt"
python "$REPO/v1/v1a3b/scripts/test_v1a3b_span_decoder.py"

mkdir -p "$WORK/manifest" "$WORK/v1a3b_assets" "$WORK/medsam_weights"

resolve_first_file() {
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

MANIFEST="${MANIFEST:-}"
if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(resolve_first_file \
    "$WORK/manifest/v1a1_dual_manifest.csv" \
    "$DRIVE/medsam-kits23/v1a2_results/reproducibility/v1a1_dual_manifest.csv" \
    "$DRIVE/medsam-kits23/v1a1_results/reproducibility/v1a1_dual_manifest.csv" \
    || true)"
fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  MANIFEST="$(find "$DRIVE/medsam-kits23" -type f -name 'v1a1_dual_manifest.csv' 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "ERROR: cannot locate v1a1_dual_manifest.csv on Drive" >&2
  exit 1
fi

LOCAL_MANIFEST="$WORK/manifest/v1a1_dual_manifest.csv"
if [[ "$MANIFEST" != "$LOCAL_MANIFEST" ]]; then
  cp "$MANIFEST" "$LOCAL_MANIFEST"
fi
MANIFEST="$LOCAL_MANIFEST"

CHECKPOINT="${V1A3A_CKPT:-}"
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(resolve_first_file \
    "$WORK/v1a3a_run/checkpoints/epoch_018.pt" \
    "$DRIVE/medsam-kits23/v1a3_results/v1a3a_run/checkpoints/epoch_018.pt" \
    || true)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$(find "$DRIVE/medsam-kits23/v1a3_results" -type f -name 'epoch_018.pt' 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: cannot locate frozen V1a.3-A epoch_018.pt" >&2
  exit 1
fi
ln -sfn "$CHECKPOINT" "$WORK/v1a3b_assets/v1a3a_epoch_018.pt"
CHECKPOINT="$WORK/v1a3b_assets/v1a3a_epoch_018.pt"

MISSING_COUNT="$(python - <<PY
from pathlib import Path
import pandas as pd
m = pd.read_csv('$MANIFEST')
print(sum(not Path(p).is_file() for p in m['npz_path'].astype(str)))
PY
)"

if [[ "$MISSING_COUNT" != "0" ]]; then
  RESTORED_ROOT="$WORK/restored_data"
  EXISTING_NPZ="$(find "$RESTORED_ROOT" -type f -name '*.npz' 2>/dev/null | head -n 1 || true)"
  if [[ -z "$EXISTING_NPZ" ]]; then
    ARCHIVE="${PROCESSED_ARCHIVE:-}"
    if [[ -z "$ARCHIVE" ]]; then
      ARCHIVE="$(resolve_first_file \
        "$DRIVE/medsam-kits23/cache/v1a1_processed_147_cases.tar" \
        "$DRIVE/medsam-kits23/cache/v1a1_processed_147_cases.tar.gz" \
        || true)"
    fi
    if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
      ARCHIVE="$(find "$DRIVE/medsam-kits23" -type f \( -name '*processed*147*.tar' -o -name '*processed*147*.tar.gz' \) 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
      echo "ERROR: manifest NPZ paths are stale and processed-data archive was not found" >&2
      exit 1
    fi
    echo "Restoring processed NPZ archive: $ARCHIVE"
    rm -rf "$RESTORED_ROOT"
    mkdir -p "$RESTORED_ROOT"
    tar -xf "$ARCHIVE" -C "$RESTORED_ROOT"
  fi

  export MANIFEST RESTORED_ROOT
  python - <<'PY'
from pathlib import Path
import os
import re
import pandas as pd

manifest_path = Path(os.environ['MANIFEST'])
root = Path(os.environ['RESTORED_ROOT'])
case_re = re.compile(r'case_\d+')
slice_re = re.compile(r'slice_(\d+)')

index = {}
for path in root.rglob('*.npz'):
    case_match = case_re.search(str(path))
    slice_match = slice_re.search(path.stem)
    if case_match and slice_match:
        index[(case_match.group(0), int(slice_match.group(1)))] = str(path.resolve())

frame = pd.read_csv(manifest_path, dtype={'case_id': str})
backup = manifest_path.with_suffix('.pre_v1a3b_restore.csv')
if not backup.exists():
    frame.to_csv(backup, index=False)

new_paths = []
missing = []
for row in frame.itertuples(index=False):
    key = (str(row.case_id), int(row.slice_idx))
    path = index.get(key)
    if path is None:
        missing.append(key)
        new_paths.append(str(row.npz_path))
    else:
        new_paths.append(path)
frame['npz_path'] = new_paths
frame.to_csv(manifest_path, index=False)
if missing:
    raise RuntimeError(f'Could not repair {len(missing)} manifest paths; examples={missing[:10]}')
print(f'Repaired {len(frame)} manifest NPZ paths under {root}')
PY
fi

python - <<PY
from pathlib import Path
import pandas as pd
m = pd.read_csv('$MANIFEST', dtype={'case_id': str})
val = m[m['split'].astype(str) == 'val']
assert val['case_id'].nunique() == 20, val['case_id'].nunique()
assert len(val) == 3644, len(val)
missing = [p for p in val['npz_path'].astype(str) if not Path(p).is_file()]
assert not missing, f'{len(missing)} validation NPZ files missing; examples={missing[:5]}'
print({'validation_cases': int(val['case_id'].nunique()), 'validation_rows': int(len(val)), 'missing_validation_npz': 0})
PY

MEDSAM_REPO="${MEDSAM_REPO:-$WORK/MedSAM}"
if [[ ! -d "$MEDSAM_REPO/segment_anything" ]]; then
  rm -rf "$MEDSAM_REPO"
  git clone -q https://github.com/bowang-lab/MedSAM.git "$MEDSAM_REPO"
fi

MEDSAM_SOURCE="${MEDSAM_CKPT_SOURCE:-}"
if [[ -z "$MEDSAM_SOURCE" ]]; then
  MEDSAM_SOURCE="$(resolve_first_file \
    "$DRIVE/medsam_vit_b.pth" \
    "$DRIVE/medsam-kits23/medsam_vit_b.pth" \
    || true)"
fi
if [[ -z "$MEDSAM_SOURCE" || ! -f "$MEDSAM_SOURCE" ]]; then
  echo "ERROR: cannot locate medsam_vit_b.pth on Drive" >&2
  exit 1
fi
ln -sfn "$MEDSAM_SOURCE" "$WORK/medsam_weights/medsam_vit_b.pth"
MEDSAM_CKPT="$WORK/medsam_weights/medsam_vit_b.pth"

ENV_FILE="$WORK/v1a3b_env.sh"
cat > "$ENV_FILE" <<EOF
export REPO="$REPO"
export WORK="$WORK"
export MANIFEST="$MANIFEST"
export V1A3A_CKPT="$CHECKPOINT"
export MEDSAM_REPO="$MEDSAM_REPO"
export MEDSAM_CKPT="$MEDSAM_CKPT"
EOF

python - <<PY
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
assert int(ckpt.get('epoch', -1)) == 18, ckpt.get('epoch')
assert 'ema_state' in ckpt
assert ckpt.get('config', {}).get('input_mode') == '2.5d'
print({'checkpoint_epoch': int(ckpt['epoch']), 'weights_available': 'ema', 'input_mode': '2.5d'})
PY

cat "$ENV_FILE"
echo
echo "V1a.3-B environment restore complete."
echo "Source it with: source $ENV_FILE"
