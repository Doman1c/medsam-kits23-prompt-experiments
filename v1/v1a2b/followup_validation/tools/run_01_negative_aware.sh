#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OUT="$FOLLOWUP/01_negative_aware"
V1A1_LOCKED="$REPO/v1/v1a1/results/recommended_final_validation_config.json"
V1A1_PREP="$OUT/v1a1_locked"
V1A1_RAW="$V1A1_PREP/val_raw_predictions.csv"

require_file "$MANIFEST"
require_file "$MEDSAM_CKPT"
require_dir "$MEDSAM_REPO"
require_file "$V1A1_LOCKED"
mkdir -p "$V1A1_PREP"

if [[ ! -f "$V1A1_RAW" ]]; then
  V1A1_CKPT="$(resolve_v1a1_checkpoint)"
  echo "V1a.1 checkpoint: $V1A1_CKPT"
  python - <<PY
import torch
path = "$V1A1_CKPT"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("epoch", -1))
print("Resolved V1a.1 checkpoint epoch:", epoch)
assert epoch == 6, f"Expected frozen V1a.1 epoch 6, found epoch {epoch}: {path}"
PY
  python "$REPO/v1/v1a1/scripts/predict_dual.py" \
    --manifest "$MANIFEST" --checkpoint "$V1A1_CKPT" \
    --output-csv "$V1A1_RAW" --split val --weights ema \
    --batch-size 32 --num-workers 2 --device cuda
fi

python "$SCRIPT_DIR/materialize_locked_predictions.py" \
  --repo-root "$REPO" --raw-predictions "$V1A1_RAW" \
  --locked-config "$V1A1_LOCKED" --output-dir "$V1A1_PREP"

python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
candidates = []
def selected_scale(summary_path):
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return float(payload["recommended_by_validation_medsam"]["expansion_scale"])
candidates.append({
    "name": "V1a.1_locked",
    "predictions": str(out / "v1a1_locked" / "recommended_predictions.csv"),
    "calibration_json": str(out / "v1a1_locked" / "recommended_calibration.json"),
    "expansion_scale": 1.20,
})
a_final = Path("/content/v1a1_work/v1a2a_presence_pareto/final")
candidates.append({
    "name": "V1a.2-A_locked",
    "predictions": str(a_final / "recommended_predictions.csv"),
    "calibration_json": str(a_final / "recommended_calibration.json"),
    "expansion_scale": selected_scale(a_final / "medsam" / "medsam_validation_summary.json"),
})
b_root = Path("/content/v1a1_work/v1a2b_validation/candidates")
for candidate_name in ("prompt_best", "fpr_best", "continuous_best", "target_recall"):
    candidate_dir = b_root / candidate_name
    if not candidate_dir.is_dir():
        continue
    candidates.append({
        "name": f"V1a.2-B_{candidate_name}",
        "predictions": str(candidate_dir / "recommended_predictions.csv"),
        "calibration_json": str(candidate_dir / "recommended_calibration.json"),
        "expansion_scale": selected_scale(candidate_dir / "medsam" / "medsam_validation_summary.json"),
    })
path = out / "negative_aware_candidate_manifest.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"candidates": candidates}, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"candidates": candidates}, indent=2))
PY

python "$SCRIPT_DIR/negative_aware_medsam_compare.py" \
  --repo-root "$REPO" \
  --candidate-manifest "$OUT/negative_aware_candidate_manifest.json" \
  --medsam-repo "$MEDSAM_REPO" --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$OUT" --target-labels 1 --device cuda

mkdir -p /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/01_negative_aware
rsync -a --delete "$OUT/" /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/01_negative_aware/
echo "TEST 1 COMPLETE: $OUT"
