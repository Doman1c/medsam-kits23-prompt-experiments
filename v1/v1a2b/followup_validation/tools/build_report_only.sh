#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python \
  "$SCRIPT_DIR/build_followup_report.py" \
  --root /content/v1a1_work/v1a2_followup_validation
