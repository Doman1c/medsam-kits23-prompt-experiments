#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
MANIFEST="${MANIFEST:-$WORK/manifest/v1a1_dual_manifest.csv}"
AUDIT_DIR="${AUDIT_DIR:-$WORK/v1a3a_presence_audit}"
RUN_DIR="${RUN_DIR:-$WORK/v1a3a_run}"
DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/medsam-kits23/v1a3_results/v1a3a_run}"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing file: $1"; exit 1; }
}

resolve_v1a2b_epoch14() {
  local candidates=(
    "$WORK/v1a2b_run/checkpoints/epoch_014.pt"
    "/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2b_run/checkpoints/epoch_014.pt"
    "/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2b_run/last.pt"
  )
  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  local found
  found="$(find \
    "$WORK" \
    /content/drive/MyDrive/medsam-kits23 \
    -type f -name 'epoch_014.pt' -path '*v1a2b*' \
    2>/dev/null | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  echo "ERROR: unable to resolve V1a.2-B epoch_014.pt" >&2
  return 1
}

require_file "$MANIFEST"
V1A2B_CKPT="${V1A2B_CKPT:-$(resolve_v1a2b_epoch14)}"
require_file "$V1A2B_CKPT"

TRAIN_PRED="$AUDIT_DIR/train_predictions.csv"
HARD_NEGATIVE="$AUDIT_DIR/train/hard_negative_pool.csv"

mkdir -p "$AUDIT_DIR" "$RUN_DIR" "$DRIVE_OUT"

echo "Repository: $REPO"
echo "Manifest: $MANIFEST"
echo "V1a.2-B checkpoint: $V1A2B_CKPT"

if [[ ! -f "$TRAIN_PRED" ]]; then
  python "$REPO/v1/v1a2b/scripts/predict_v1a2b.py" \
    --manifest "$MANIFEST" \
    --checkpoint "$V1A2B_CKPT" \
    --output-csv "$TRAIN_PRED" \
    --split train \
    --weights ema \
    --batch-size 32 \
    --num-workers 2 \
    --presence-threshold 0.5 \
    --max-gap 2 \
    --device cuda
fi

python "$REPO/v1/v1a3/scripts/audit_presence_errors.py" \
  --repo-root "$REPO" \
  --manifest "$MANIFEST" \
  --predictions "$TRAIN_PRED" \
  --output-dir "$AUDIT_DIR/train" \
  --split train \
  --variant smooth \
  --left-threshold 0.01 \
  --right-threshold 0.01 \
  --max-gap 2 \
  --hard-negative-min-score 0.20 \
  --hard-negative-top-k-per-case 32

require_file "$HARD_NEGATIVE"

python "$REPO/v1/v1a3/scripts/test_v1a3_sampler.py" \
  --repo-root "$REPO" \
  --manifest "$MANIFEST" \
  --hard-negative-csv "$HARD_NEGATIVE" \
  --output-dir "$AUDIT_DIR/sampler_gate" \
  --negative-fraction 0.45 \
  --mined-negative-share 0.45 \
  --boundary-negative-share 0.35 \
  --near-negative-share 0.15 \
  --far-negative-share 0.05 \
  --boundary-radius 8 \
  --unilateral-positive-boost 1.50 \
  --seed 42

python "$REPO/v1/v1a3/scripts/train_v1a3a.py" \
  --manifest "$MANIFEST" \
  --hard-negative-csv "$HARD_NEGATIVE" \
  --output-dir "$RUN_DIR" \
  --backup-dir "$DRIVE_OUT" \
  --epochs 40 \
  --batch-size 32 \
  --num-workers 2 \
  --input-mode 2.5d \
  --negative-fraction 0.45 \
  --mined-negative-share 0.45 \
  --boundary-negative-share 0.35 \
  --near-negative-share 0.15 \
  --far-negative-share 0.05 \
  --boundary-radius 8 \
  --unilateral-positive-boost 1.50 \
  --device cuda

echo
echo "V1a.3-A complete"
echo "Run directory: $RUN_DIR"
echo "Drive backup: $DRIVE_OUT"
