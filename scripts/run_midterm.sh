#!/usr/bin/env bash
# Full mid-project pipeline (run after scripts/download_data.sh).
#   1. sanity check: held-out COCO images unseen by NegCLIP
#   2. encode all benchmarks with CLIP and NegCLIP (cached; ~1.2 h/model on CPU)
#   3. Approach 01: gap metrics + PCA figures
#   4. Approach 02 + 04: shift sweep, evaluation, plots
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODELS="${MODELS:-clip negclip}"

$PY scripts/check_overlap.py
$PY scripts/embed_all.py --models $MODELS
$PY scripts/measure_gap.py --models $MODELS
$PY scripts/shift_eval.py --models $MODELS
$PY scripts/plot_shift.py
