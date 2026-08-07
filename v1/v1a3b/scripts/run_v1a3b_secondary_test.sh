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

OUT="${OUT:-$WORK/v1a3b_secondary_test}"
DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/medsam-kits23/v1a3b_results/secondary_test}"

LOCK="$REPO/v1/v1a3b/FROZEN_LOCK.json"
MATERIALIZE="$REPO/v1/v1a3b/scripts/materialize_v1a3b_secondary_test.py"
PREDICT="$REPO/v1/v1a2b/scripts/predict_v1a2b.py"
PROMPT_EVAL="$REPO/v1/v1a1/scripts/evaluate_dual_prompts.py"
NEGATIVE_SOURCE="$REPO/v1/v1a2b/followup_validation/tools/negative_aware_medsam_compare.py"
NEGATIVE_TEST="$WORK/v1a3b_negative_aware_secondary_test.py"

EXPECTED_CKPT_SHA="f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2"

for path in \
  "$LOCK" \
  "$MATERIALIZE" \
  "$PREDICT" \
  "$PROMPT_EVAL" \
  "$NEGATIVE_SOURCE" \
  "$MANIFEST" \
  "$V1A3A_CKPT" \
  "$MEDSAM_CKPT"
do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

[[ -d "$MEDSAM_REPO/segment_anything" ]] || {
  echo "ERROR: invalid MedSAM repo: $MEDSAM_REPO" >&2
  exit 1
}

ACTUAL_CKPT_SHA="$(sha256sum "$V1A3A_CKPT" | awk '{print $1}')"
if [[ "$ACTUAL_CKPT_SHA" != "$EXPECTED_CKPT_SHA" ]]; then
  echo "ERROR: frozen V1a.3-A checkpoint SHA mismatch" >&2
  echo "Expected: $EXPECTED_CKPT_SHA" >&2
  echo "Actual:   $ACTUAL_CKPT_SHA" >&2
  exit 1
fi

python - <<PY
import json
from pathlib import Path

lock = json.loads(Path("$LOCK").read_text(encoding="utf-8"))
winner = lock["validation_selection"]
decoder = winner["decoder"]

assert lock["status"] == "validation_frozen_before_secondary_test"
assert winner["winner"] == "b1_low_fpr"
assert decoder["mode"] == "single_span"
assert abs(float(decoder["left_threshold"]) - 0.0035) < 1e-12
assert abs(float(decoder["right_threshold"]) - 0.04) < 1e-12
assert int(decoder["baseline_max_gap"]) == 4
assert int(decoder["boundary_pad"]) == 1
assert abs(float(winner["expansion_scale"]) - 1.20) < 1e-12

print("Frozen V1a.3-B secondary-test lock verified.")
PY

rm -rf "$OUT"
mkdir -p "$OUT" "$DRIVE_OUT"
cp "$LOCK" "$OUT/FROZEN_LOCK.json"

BASE_PRED="${BASE_TEST_PREDICTIONS:-}"

if [[ -z "$BASE_PRED" ]]; then
  for path in \
    "$WORK/v1a3a_test_final/test_base_predictions.csv" \
    "/content/drive/MyDrive/medsam-kits23/v1a3_results/v1a3a_test_final/test_base_predictions.csv"
  do
    if [[ -f "$path" ]]; then
      BASE_PRED="$path"
      break
    fi
  done
fi

if [[ -z "$BASE_PRED" || ! -f "$BASE_PRED" ]]; then
  BASE_PRED="$OUT/test_base_predictions.csv"
  echo "Existing V1a.3-A test base predictions not found; regenerating the same frozen epoch-18 EMA outputs."
  python "$PREDICT" \
    --manifest "$MANIFEST" \
    --checkpoint "$V1A3A_CKPT" \
    --output-csv "$BASE_PRED" \
    --split test \
    --weights ema \
    --batch-size 32 \
    --num-workers 2 \
    --presence-threshold 0.5 \
    --max-gap 0 \
    --device cuda
else
  echo "Reusing frozen V1a.3-A test base predictions: $BASE_PRED"
fi

python - <<PY
from pathlib import Path
import pandas as pd

manifest = pd.read_csv("$MANIFEST", dtype={"case_id": str})
test = manifest[manifest["split"].astype(str) == "test"].copy()
pred = pd.read_csv("$BASE_PRED", dtype={"case_id": str})

assert test["case_id"].nunique() == 27
assert len(test) == 4277
assert len(pred) == 4277

test_keys = set(zip(test["case_id"].astype(str), test["slice_idx"].astype(int)))
pred_keys = set(zip(pred["case_id"].astype(str), pred["slice_idx"].astype(int)))
assert pred_keys == test_keys

missing = [p for p in pred["npz_path"].astype(str) if not Path(p).is_file()]
assert not missing, f"{len(missing)} prediction NPZ paths are missing; examples={missing[:5]}"

print({
    "test_cases": int(test["case_id"].nunique()),
    "test_rows": int(len(test)),
    "base_prediction_rows": int(len(pred)),
    "missing_npz": 0,
})
PY

CANDIDATE_SET="$OUT/candidate_set"

python "$MATERIALIZE" \
  --predictions "$BASE_PRED" \
  --manifest "$MANIFEST" \
  --output-dir "$CANDIDATE_SET"

for candidate in b0_locked_baseline b1_low_fpr_frozen; do
  CANDIDATE_DIR="$CANDIDATE_SET/candidates/$candidate"
  rm -rf "$CANDIDATE_DIR/prompt_metrics"
  python "$PROMPT_EVAL" \
    --predictions "$CANDIDATE_DIR/predictions.csv" \
    --output-dir "$CANDIDATE_DIR/prompt_metrics" \
    --variant smooth \
    --expansion-scale 1.20 \
    --target-labels 1
done

export NEGATIVE_SOURCE NEGATIVE_TEST
python - <<'PY'
from pathlib import Path
import os

source = Path(os.environ["NEGATIVE_SOURCE"])
destination = Path(os.environ["NEGATIVE_TEST"])
text = source.read_text(encoding="utf-8")

replacements = {
    '"status": "validation_only_negative_aware_medsam"':
        '"status": "v1a3b_locked_secondary_test_negative_aware_medsam"',
    '"development_split": "validation"':
        '"evaluation_split": "previously_exposed_test"',
    '"test_set_used": False':
        '"test_set_used": True',
}

for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"Expected patch target not found: {old}")
    text = text.replace(old, new)

destination.write_text(text, encoding="utf-8")
print("Created:", destination)
PY

python -m py_compile "$NEGATIVE_TEST"

NEGATIVE_OUT="$OUT/negative_aware_medsam"

python "$NEGATIVE_TEST" \
  --repo-root "$REPO" \
  --candidate-manifest "$CANDIDATE_SET/candidate_manifest.json" \
  --medsam-repo "$MEDSAM_REPO" \
  --checkpoint "$MEDSAM_CKPT" \
  --output-dir "$NEGATIVE_OUT" \
  --target-labels 1 \
  --device cuda

export OUT
python - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd

root = Path(os.environ["OUT"])
table = pd.read_csv(root / "negative_aware_medsam" / "negative_aware_candidate_comparison.csv")
table = table.set_index("candidate")

required = {"b0_locked_baseline", "b1_low_fpr_frozen"}
if not required.issubset(table.index):
    raise RuntimeError(f"Missing expected candidates: {required - set(table.index)}")

b0 = table.loc["b0_locked_baseline"]
b1 = table.loc["b1_low_fpr_frozen"]

expected_b0 = {
    "all_slice_dice_mean": 0.747545878458055,
    "positive_slice_dice_mean": 0.7233743026900277,
    "negative_slice_specificity": 0.7602568676418123,
    "case_volume_dice_mean": 0.7995902880973497,
}
for key, expected in expected_b0.items():
    actual = float(b0[key])
    if abs(actual - expected) > 1e-6:
        raise RuntimeError(
            f"B0 reproduction mismatch for {key}: expected={expected}, actual={actual}"
        )

metric_keys = [
    "all_slice_dice_mean",
    "positive_slice_dice_mean",
    "positive_slice_case_mean_dice",
    "positive_slice_detection_recall",
    "negative_slice_specificity",
    "negative_false_positive_mask_rate",
    "case_volume_dice_mean",
]

summary = {
    "experiment": "V1a.3-B",
    "status": "locked_secondary_test_complete",
    "test_interpretation": (
        "Secondary locked evaluation on the same 27-case test split previously exposed "
        "during V1a.3-A; not a pristine first-look test."
    ),
    "selection_performed_on_test": False,
    "frozen_winner": {
        "name": "b1_low_fpr_frozen",
        "decoder": {
            "mode": "single_span",
            "left_threshold": 0.0035,
            "right_threshold": 0.04,
            "baseline_max_gap": 4,
            "boundary_pad": 1,
            "expansion_scale": 1.20
        }
    },
    "b0": {key: float(b0[key]) for key in metric_keys},
    "b1": {key: float(b1[key]) for key in metric_keys},
    "b1_minus_b0": {key: float(b1[key] - b0[key]) for key in metric_keys},
    "post_test_tuning_allowed": False
}

(root / "secondary_test_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

comparison = pd.DataFrame([
    {"candidate": "b0_locked_baseline", **summary["b0"]},
    {"candidate": "b1_low_fpr_frozen", **summary["b1"]},
    {"candidate": "b1_minus_b0", **summary["b1_minus_b0"]},
])
comparison.to_csv(root / "secondary_test_comparison.csv", index=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

rsync -a "$OUT/" "$DRIVE_OUT/"
sync

echo
echo "=================================================="
echo "V1a.3-B locked secondary test complete"
echo "Content: $OUT"
echo "Drive:   $DRIVE_OUT"
echo "=================================================="
