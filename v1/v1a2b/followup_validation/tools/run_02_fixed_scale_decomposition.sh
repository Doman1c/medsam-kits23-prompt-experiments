#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OUT="$FOLLOWUP/02_fixed_scale"
CONFIGS="$OUT/configs"
EVALUATOR="$SCRIPT_DIR/validation/evaluate_center_size_generic.py"
require_file "$MANIFEST"
require_file "$MEDSAM_CKPT"
require_dir "$MEDSAM_REPO"
require_file "$EVALUATOR"

V1A1_PRED="$FOLLOWUP/01_negative_aware/v1a1_locked/recommended_predictions.csv"
V1A2A_PRED="$WORK/v1a2a_presence_pareto/final/recommended_predictions.csv"
V1A2B_PRED="$WORK/v1a2b_validation/final/recommended_predictions.csv"
require_file "$V1A1_PRED"
require_file "$V1A2A_PRED"
require_file "$V1A2B_PRED"
mkdir -p "$CONFIGS"

python - <<PY
import json
from pathlib import Path
repo = Path("$REPO")
work = Path("$WORK")
configs = Path("$CONFIGS")
sources = {
    "V1a.1": repo / "v1/v1a1/results/recommended_final_validation_config.json",
    "V1a.2-A": work / "v1a2a_presence_pareto/final/recommended_final_validation_config.json",
    "V1a.2-B": work / "v1a2b_validation/final/recommended_final_validation_config.json",
}
for name, source in sources.items():
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["development_split"] = "validation"
    payload["test_set_used"] = False
    payload["experiment"] = name
    payload["expansion_scale"] = 1.20
    destination = configs / (name.lower().replace(".", "").replace("-", "") + "_fixed_120.json")
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(name, "->", destination)
PY

rm -rf "$OUT/v1a1" "$OUT/v1a2a" "$OUT/v1a2b" "$OUT/comparison"

python "$EVALUATOR" --experiment "V1a.1-fixed-1.20" --repo-root "$REPO" --manifest "$MANIFEST" --predictions "$V1A1_PRED" --final-config "$CONFIGS/v1a1_fixed_120.json" --medsam-repo "$MEDSAM_REPO" --checkpoint "$MEDSAM_CKPT" --output-dir "$OUT/v1a1" --target-labels 1 --expected-val-cases 20 --device cuda
python "$EVALUATOR" --experiment "V1a.2-A-fixed-1.20" --repo-root "$REPO" --manifest "$MANIFEST" --predictions "$V1A2A_PRED" --final-config "$CONFIGS/v1a2a_fixed_120.json" --medsam-repo "$MEDSAM_REPO" --checkpoint "$MEDSAM_CKPT" --output-dir "$OUT/v1a2a" --target-labels 1 --expected-val-cases 20 --device cuda
python "$EVALUATOR" --experiment "V1a.2-B-fixed-1.20" --repo-root "$REPO" --manifest "$MANIFEST" --predictions "$V1A2B_PRED" --final-config "$CONFIGS/v1a2b_fixed_120.json" --medsam-repo "$MEDSAM_REPO" --checkpoint "$MEDSAM_CKPT" --output-dir "$OUT/v1a2b" --target-labels 1 --expected-val-cases 20 --device cuda

python "$SCRIPT_DIR/compare_fixed_scale_decomposition.py" \
  --model "V1a.1::$OUT/v1a1/center_size_decomposition_summary.json" \
  --model "V1a.2-A::$OUT/v1a2a/center_size_decomposition_summary.json" \
  --model "V1a.2-B::$OUT/v1a2b/center_size_decomposition_summary.json" \
  --output-dir "$OUT/comparison"

mkdir -p /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/02_fixed_scale
rsync -a --delete "$OUT/" /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/02_fixed_scale/
echo "TEST 2 COMPLETE: $OUT"
