#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT:?Set EXPERIMENT, e.g. V1a.2-A}"
: "${PREDICT_SCRIPT:?Set PREDICT_SCRIPT}"
: "${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT}"
: "${WORK:?Set WORK}"

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V1A1="$REPO/v1/v1a1"
MANIFEST="${MANIFEST:-/content/v1a1_work/manifest/v1a1_dual_manifest.csv}"
MEDSAM_REPO="${MEDSAM_REPO:-/content/v1a1_work/MedSAM}"
MEDSAM_CKPT="${MEDSAM_CKPT:-/content/drive/MyDrive/medsam_vit_b.pth}"
DRIVE_OUT="${DRIVE_OUT:-}"

RAW="$WORK/raw/val_base_predictions.csv"
SWEEP="$WORK/presence_sweep"
CANDIDATES="$WORK/candidates"
SELECTION="$WORK/selection"
FINAL="$WORK/final"
DECOMP="$WORK/center_size_decomposition"

for path in \
  "$PREDICT_SCRIPT" \
  "$MODEL_CHECKPOINT" \
  "$MANIFEST" \
  "$MEDSAM_CKPT" \
  "$V1A1/scripts/calibrate_presence_smoothing.py" \
  "$V1A1/scripts/medsam_validation_sweep.py"
do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing path: $path"
    exit 1
  fi
done

if [[ ! -d "$MEDSAM_REPO/segment_anything" ]]; then
  echo "ERROR: invalid MedSAM repo: $MEDSAM_REPO"
  exit 1
fi

mkdir -p "$WORK/raw" "$SWEEP" "$CANDIDATES" "$SELECTION" "$FINAL" "$DECOMP"

python "$TOOLS_DIR/validation/patch_medsam_validation.py" \
  --script "$V1A1/scripts/medsam_validation_sweep.py"
python -m py_compile "$V1A1/scripts/medsam_validation_sweep.py"

echo "===== 1/7 Export validation predictions: $EXPERIMENT ====="
python "$PREDICT_SCRIPT" \
  --manifest "$MANIFEST" \
  --checkpoint "$MODEL_CHECKPOINT" \
  --output-csv "$RAW" \
  --split val \
  --weights ema \
  --batch-size 32 \
  --device cuda

echo "===== 2/7 Wide presence calibration sweep ====="
rm -rf "$SWEEP"
mkdir -p "$SWEEP"
python "$V1A1/scripts/calibrate_presence_smoothing.py" \
  --predictions "$RAW" \
  --output-dir "$SWEEP" \
  --left-thresholds 0.01,0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55 \
  --right-thresholds 0.01,0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55 \
  --max-gaps 0,1,2 \
  --expansion-scale 1.15 \
  --target-labels 1 \
  --min-overall-recall 0.95 \
  --min-side-recall 0.90

python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$SWEEP/recommended_calibration.json").read_text(encoding="utf-8"))
assert payload["constraints_met"], "No calibration met the locked recall constraints."
print("Recall constraints: PASSED")
PY

echo "===== 3/7 Select diverse feasible presence candidates ====="
rm -rf "$CANDIDATES"
mkdir -p "$CANDIDATES"
python "$TOOLS_DIR/validation/select_presence_candidates.py" \
  --v1a1-root "$V1A1" \
  --sweep-csv "$SWEEP/presence_smoothing_sweep.csv" \
  --base-predictions "$RAW" \
  --output-dir "$CANDIDATES" \
  --min-overall-recall 0.95 \
  --min-side-recall 0.90 \
  --target-recall 0.95

echo "===== 4/7 Full MedSAM validation for each candidate ====="
mapfile -t CANDIDATE_DIRS < <(
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$CANDIDATES/candidate_manifest.json").read_text(encoding="utf-8"))
for candidate in payload["candidates"]:
    print(candidate["directory"])
PY
)

for candidate_dir in "${CANDIDATE_DIRS[@]}"; do
  echo "--- Candidate: $(basename "$candidate_dir") ---"
  rm -rf "$candidate_dir/medsam"
  mkdir -p "$candidate_dir/medsam"
  python "$V1A1/scripts/medsam_validation_sweep.py" \
    --predictions "$candidate_dir/recommended_predictions.csv" \
    --calibration-json "$candidate_dir/recommended_calibration.json" \
    --medsam-repo "$MEDSAM_REPO" \
    --checkpoint "$MEDSAM_CKPT" \
    --output-dir "$candidate_dir/medsam" \
    --scales 1.10,1.15,1.20,1.25 \
    --target-labels 1 \
    --device cuda
done

echo "===== 5/7 Select presence candidate by MedSAM Dice ====="
rm -rf "$SELECTION"
mkdir -p "$SELECTION"
python "$TOOLS_DIR/validation/compare_presence_candidates.py" \
  --candidate-manifest "$CANDIDATES/candidate_manifest.json" \
  --output-dir "$SELECTION"

echo "===== 6/7 Materialize locked validation winner ====="
rm -rf "$FINAL"
mkdir -p "$FINAL"
python "$TOOLS_DIR/validation/finalize_presence_candidate.py" \
  --experiment "$EXPERIMENT" \
  --winner-json "$SELECTION/presence_candidate_winner.json" \
  --checkpoint "$MODEL_CHECKPOINT" \
  --output-dir "$FINAL"

echo "===== 7/7 Four-mode center-size decomposition ====="
rm -rf "$DECOMP"
mkdir -p "$DECOMP"
python "$TOOLS_DIR/validation/evaluate_center_size_generic.py" \
  --experiment "$EXPERIMENT" \
  --repo-root "$REPO" \
  --manifest "$MANIFEST" \
  --predictions "$FINAL/recommended_predictions.csv" \
  --final-config "$FINAL/recommended_final_validation_config.json" \
  --medsam-repo "$MEDSAM_REPO" \
  --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$DECOMP" \
  --target-labels 1 \
  --expected-val-cases 20 \
  --device cuda

if [[ -n "$DRIVE_OUT" ]]; then
  mkdir -p "$DRIVE_OUT"
  rsync -a --delete "$WORK/" "$DRIVE_OUT/"
fi

echo "$EXPERIMENT PRESENCE-PARETO VALIDATION COMPLETED"
echo "Final config: $FINAL/recommended_final_validation_config.json"
echo "Decomposition: $DECOMP/center_size_decomposition_summary.json"
