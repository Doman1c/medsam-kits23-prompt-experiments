#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/medsam-kits23-prompt-experiments}"
WORK="${WORK:-/content/v1a1_work}"
ENV_FILE="$WORK/v1a3b_env.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Run restore_v1a3b_colab.sh first." >&2
  exit 1
fi
source "$ENV_FILE"

RAW="${RAW:-$WORK/v1a3b_validation/val_epoch18_base_predictions.csv}"
STAGE1B="${STAGE1B:-$WORK/v1a3b_stage1b/span_sweep}"
OUT="${OUT:-$WORK/v1a3b_stage2_medsam}"
DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/medsam-kits23/v1a3b_results/stage2_medsam}"

PREPARE="$REPO/v1/v1a3b/scripts/prepare_v1a3b_stage2_candidates.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
NEGATIVE_AWARE="$REPO/v1/v1a2b/followup_validation/tools/negative_aware_medsam_compare.py"

for path in \
  "$MANIFEST" \
  "$RAW" \
  "$STAGE1B/span_decoder_stage1b_sweep.csv" \
  "$PREPARE" \
  "$PROMPT_EVAL" \
  "$NEGATIVE_AWARE" \
  "$MEDSAM_CKPT"
do
  [[ -e "$path" ]] || {
    echo "ERROR: missing $path" >&2
    exit 1
  }
done

[[ -d "$MEDSAM_REPO/segment_anything" ]] || {
  echo "ERROR: invalid MedSAM repository: $MEDSAM_REPO" >&2
  exit 1
}

rm -rf "$OUT"
mkdir -p "$OUT" "$DRIVE_OUT"

python "$PREPARE" \
  --predictions "$RAW" \
  --manifest "$MANIFEST" \
  --sweep-csv "$STAGE1B/span_decoder_stage1b_sweep.csv" \
  --output-dir "$OUT/candidate_set" \
  --expansion-scale 1.20

mapfile -t CANDIDATE_DIRS < <(
  python - <<PY
import json
from pathlib import Path
payload = json.loads(
    Path("$OUT/candidate_set/candidate_manifest.json").read_text(encoding="utf-8")
)
for item in payload["candidates"]:
    print(item["directory"])
PY
)

for candidate_dir in "${CANDIDATE_DIRS[@]}"; do
  rm -rf "$candidate_dir/prompt_metrics"
  python "$PROMPT_EVAL" \
    --predictions "$candidate_dir/recommended_predictions.csv" \
    --output-dir "$candidate_dir/prompt_metrics" \
    --variant smooth \
    --expansion-scale 1.20 \
    --target-labels 1
done

python "$NEGATIVE_AWARE" \
  --repo-root "$REPO" \
  --candidate-manifest "$OUT/candidate_set/candidate_manifest.json" \
  --medsam-repo "$MEDSAM_REPO" \
  --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$OUT/negative_aware_medsam" \
  --target-labels 1 \
  --device cuda

python - <<PY
from pathlib import Path
import pandas as pd

out = Path("$OUT")
presence = pd.read_csv(out / "candidate_set/stage2_presence_candidates.csv")
medsam = pd.read_csv(out / "negative_aware_medsam/negative_aware_candidate_comparison.csv")
if "candidate" not in medsam.columns:
    raise RuntimeError(f"Expected candidate column, got {list(medsam.columns)}")
merged = presence.merge(
    medsam,
    left_on="name",
    right_on="candidate",
    how="left",
    validate="one_to_one",
)
merged.to_csv(out / "stage2_combined_comparison.csv", index=False)
print(merged.to_string(index=False))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "V1a.3-B Stage 2 MedSAM complete."
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
