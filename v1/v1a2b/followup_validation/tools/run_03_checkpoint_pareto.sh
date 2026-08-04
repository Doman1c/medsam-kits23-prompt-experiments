#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OUT="$FOLLOWUP/03_checkpoint_pareto"
PIPELINE="$SCRIPT_DIR/validation/run_presence_pareto_validation.sh"
PREDICT="$REPO/v1/v1a2b/scripts/predict_v1a2b.py"
require_file "$MANIFEST"
require_file "$MEDSAM_CKPT"
require_dir "$MEDSAM_REPO"
require_file "$PIPELINE"
require_file "$PREDICT"
mkdir -p "$OUT"

for epoch in 4 5 11 14; do
  name="$(printf 'epoch_%03d' "$epoch")"
  checkpoint="$(resolve_epoch_checkpoint "$epoch")"
  echo "Checkpoint Pareto: $name"
  EXPERIMENT="V1a.2-B-$name" \
  PREDICT_SCRIPT="$PREDICT" \
  MODEL_CHECKPOINT="$checkpoint" \
  WORK="$OUT/$name" \
  REPO="$REPO" \
  MANIFEST="$MANIFEST" \
  MEDSAM_REPO="$MEDSAM_REPO" \
  MEDSAM_CKPT="$MEDSAM_CKPT" \
  DRIVE_OUT="/content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/03_checkpoint_pareto/$name" \
  bash "$PIPELINE"
done

python - <<PY
import json
from pathlib import Path
root = Path("$OUT")
candidates = []
for epoch in (4, 5, 11, 14):
    name = f"epoch_{epoch:03d}"
    final_dir = root / name / "final"
    config = json.loads((final_dir / "recommended_final_validation_config.json").read_text(encoding="utf-8"))
    candidates.append({
        "name": name,
        "predictions": str(final_dir / "recommended_predictions.csv"),
        "calibration_json": str(final_dir / "recommended_calibration.json"),
        "expansion_scale": float(config["expansion_scale"]),
    })
path = root / "checkpoint_winner_candidate_manifest.json"
path.write_text(json.dumps({"candidates": candidates}, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"candidates": candidates}, indent=2))
PY

python "$SCRIPT_DIR/negative_aware_medsam_compare.py" \
  --repo-root "$REPO" \
  --candidate-manifest "$OUT/checkpoint_winner_candidate_manifest.json" \
  --medsam-repo "$MEDSAM_REPO" --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$OUT/negative_aware_winners" --target-labels 1 --device cuda

python "$SCRIPT_DIR/compare_checkpoint_pareto.py" \
  --checkpoint-root "$OUT" \
  --negative-aware-summary "$OUT/negative_aware_winners/negative_aware_summary.json" \
  --output-dir "$OUT/comparison" --epochs 4,5,11,14

mkdir -p /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/03_checkpoint_pareto
rsync -a "$OUT/" /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/03_checkpoint_pareto/
echo "TEST 3 COMPLETE: $OUT"
