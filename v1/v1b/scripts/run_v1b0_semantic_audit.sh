#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
OUT="${OUT:-/content/v1b0_semantic_audit}"
DRIVE="${DRIVE:-/content/drive/MyDrive}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE/medsam-kits23/v1b_results/v1b0_semantic_audit}"

[[ -d "$REPO/.git" ]] || { echo "ERROR: repo missing: $REPO" >&2; exit 1; }

# V1b.0 only needs the inherited development manifest and processed NPZ files.
# Reuse the already-tested V1a restore if the current Colab runtime does not
# already contain them. This does not run any V1b training or MedSAM evaluation.
if [[ ! -f "$WORK/v1a6_env.sh" ]]; then
  bash "$REPO/v1/v1a6/scripts/restore_v1a6_colab.sh"
fi
source "$WORK/v1a6_env.sh"

[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT" "$DRIVE_OUT"

python "$REPO/v1/v1b/scripts/audit_v1b_semantic_targets.py" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT" \
  --allowed-splits train,val \
  --min-component-area 5

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1b.0 semantic-target audit complete."
echo "Local: $OUT"
echo "Drive: $DRIVE_OUT"
echo "No training, no MedSAM evaluation, no test access."
