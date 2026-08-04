#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_01_negative_aware.sh"
bash "$SCRIPT_DIR/run_02_fixed_scale_decomposition.sh"
bash "$SCRIPT_DIR/run_03_checkpoint_pareto.sh"
python "$SCRIPT_DIR/build_followup_report.py" --root /content/v1a1_work/v1a2_followup_validation
mkdir -p /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/final_report
rsync -a --delete /content/v1a1_work/v1a2_followup_validation/final_report/ /content/drive/MyDrive/medsam-kits23/v1a2_results/followup_validation/final_report/
echo "ALL THREE FOLLOW-UP VALIDATIONS COMPLETED"
