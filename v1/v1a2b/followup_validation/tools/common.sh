#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
TOOLS="${TOOLS:-/content/v1a2_followup_validation_tools}"
FOLLOWUP="${FOLLOWUP:-$WORK/v1a2_followup_validation}"
MANIFEST="${MANIFEST:-$WORK/manifest/v1a1_dual_manifest.csv}"
MEDSAM_REPO="${MEDSAM_REPO:-$WORK/MedSAM}"
MEDSAM_CKPT="${MEDSAM_CKPT:-/content/drive/MyDrive/medsam_vit_b.pth}"
mkdir -p "$FOLLOWUP"

require_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1"; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "ERROR: missing directory: $1"; exit 1; }; }

resolve_v1a1_checkpoint() {
  local candidates=(
    "$WORK/v1a1_run/best_continuous.pt"
    "$WORK/run/best_continuous.pt"
    "/content/drive/MyDrive/medsam-kits23/v1a1_results/run/best_continuous.pt"
    "/content/drive/MyDrive/medsam-kits23/v1a1_results/best_continuous.pt"
  )
  for path in "${candidates[@]}"; do
    [[ -f "$path" ]] && { echo "$path"; return 0; }
  done
  local found
  found="$(find /content/drive/MyDrive/medsam-kits23 "$WORK" -type f -name best_continuous.pt -path '*v1a1*' 2>/dev/null | head -n 1 || true)"
  [[ -n "$found" ]] && { echo "$found"; return 0; }
  echo "ERROR: unable to resolve V1a.1 best_continuous.pt" >&2
  return 1
}

resolve_epoch_checkpoint() {
  local epoch="$1" name
  name="$(printf 'epoch_%03d.pt' "$epoch")"
  local candidates=(
    "$WORK/v1a2b_run/checkpoints/$name"
    "/content/drive/MyDrive/medsam-kits23/v1a2_results/v1a2b_run/checkpoints/$name"
  )
  for path in "${candidates[@]}"; do
    [[ -f "$path" ]] && { echo "$path"; return 0; }
  done
  echo "ERROR: unable to resolve $name" >&2
  return 1
}
