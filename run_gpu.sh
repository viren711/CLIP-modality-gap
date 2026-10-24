#!/usr/bin/env bash
# =========================================================================
#  CS 725 - modality gap x hard negatives: everything that needs the GPU.
#
#  Run from the unzipped package root:
#      bash run_gpu.sh
#  Options (environment variables):
#      GPUS="0 1 2 3"     GPUs to use; fine-tuning jobs run one per GPU   (default "0")
#      LRS="1e-5 5e-6 1e-6"  learning-rate sweep, as in the NegCLIP paper
#      SEEDS="0"          seeds per (config, lr); e.g. "0 1 2" for error bars
#      PY=/path/python    use an existing Python that has CUDA torch
#                         (otherwise a venv .venv-gpu is created)
#      INCLUDE_EMB=1      put cached embeddings (~3 GB) in the results tarball
#      TORCH_INDEX=url    torch wheel index, e.g. https://download.pytorch.org/whl/cu126
#      PRUNE_RUNS=1       after LR selection, delete checkpoints of non-selected runs
#  All caches (pip, model weights, nltk, matplotlib) stay inside this folder.
#  Every step is resumable: re-running skips finished work.
#
#  Steps
#    0  Python env               4  fine-tuning sweep (4 configs x LRS x SEEDS)
#    1  GPU check + smoke test   5  pick best LR per config -> checkpoints/
#    2  full COCO 2014 images    6  encode all benchmarks with all 7 models
#    3  leakage check            7  gap metrics, shift sweep, plots
#                                8  pack results -> results_gpu.tar.gz
#  Send back: results_gpu.tar.gz
# =========================================================================
set -euo pipefail
cd "$(dirname "$0")"
GPUS="${GPUS:-0}"
LRS="${LRS:-1e-5 5e-6 1e-6}"
SEEDS="${SEEDS:-0}"
INCLUDE_EMB="${INCLUDE_EMB:-1}"
MODELS="clip negclip negclip_gelu ft_plain ft_hardimg ft_hardcap ft_negclip"
# torch wheel index; pick the CUDA build that the server's driver supports,
# e.g. TORCH_INDEX=https://download.pytorch.org/whl/cu126 for a CUDA 12.6 driver
TORCH_INDEX="${TORCH_INDEX:-}"
mkdir -p logs/gpu

# keep every download / cache inside this folder (nothing written to ~/.cache etc.)
export XDG_CACHE_HOME="$PWD/.cache" XDG_CONFIG_HOME="$PWD/.cache/config" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" PIP_CACHE_DIR="$PWD/.cache/pip"
export NLTK_DATA="$PWD/.cache/nltk_data"
mkdir -p "$MPLCONFIGDIR" "$NLTK_DATA"
step() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

# 0. Python environment ------------------------------------------------------
step "0  Python environment"
if [[ -z "${PY:-}" ]]; then
    if [[ ! -x .venv-gpu/bin/python ]]; then
        python3 -m venv .venv-gpu 2>/dev/null || {
            rm -rf .venv-gpu
            python3 -m venv --without-pip .venv-gpu
            curl -sS https://bootstrap.pypa.io/get-pip.py | .venv-gpu/bin/python
        }
        .venv-gpu/bin/pip install -q --upgrade pip
        # default PyPI wheels include CUDA on Linux; TORCH_INDEX overrides the build
        .venv-gpu/bin/pip install -q torch torchvision ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
    fi
    PY="$PWD/.venv-gpu/bin/python"
fi
$PY -m pip install -q -r requirements.txt
$PY -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True)" || true

# 1. GPU check + short smoke test (catches OOM before the long sweep) ---------
step "1  GPU check"
$PY - <<'EOF'
import torch, open_clip
assert torch.cuda.is_available(), "CUDA not available - check the torch install / driver"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
print("torch", torch.__version__, "| open_clip", open_clip.__version__)
EOF
{ nvidia-smi || true; $PY -m pip freeze; } > logs/gpu/env.txt 2>&1

# 2. Full COCO 2014 images (train2014 13 GB + rest of val2014) ----------------
step "2  COCO 2014 images"
bash scripts/gpu/prepare_coco_full.sh 2>&1 | tee logs/gpu/prepare_coco.log

if [[ ! -f runs/_smoke_ok ]]; then
    step "1b smoke test: 20 steps of the heaviest config (NegCLIP, batch 256)"
    CUDA_VISIBLE_DEVICES="${GPUS%% *}" $PY scripts/train_finetune.py --name _smoke --hard-images --hard-captions \
        --epochs 1 --max-steps 20 --out-dir runs/_smoke 2>&1 | tail -5
    rm -rf runs/_smoke && touch runs/_smoke_ok
fi

# 3. Leakage check --------------------------------------------------------------
step "3  leakage check"
$PY scripts/check_overlap.py | tee logs/gpu/check_overlap.log

# 4. Fine-tuning sweep ------------------------------------------------------------
step "4  fine-tuning sweep: GPUs [$GPUS] LRs [$LRS] seeds [$SEEDS]"
$PY scripts/gpu/run_sweep.py --gpus $GPUS --lrs $LRS --seeds $SEEDS 2>&1 | tee -a logs/gpu/sweep.log

# 5. Select best LR per config -> checkpoints/<config>.pt --------------------------
step "5  select best learning rate per config"
$PY scripts/gpu/select_best.py ${PRUNE_RUNS:+--prune} | tee logs/gpu/select_best.log

# 6. Encode all benchmarks with all models --------------------------------------
step "6  encode benchmarks: $MODELS"
CUDA_VISIBLE_DEVICES="${GPUS%% *}" $PY scripts/embed_all.py --models $MODELS --workers 8 2>&1 \
    | grep -v "it/s]" | tee logs/gpu/embed.log

# 7. Gap metrics, shift sweep, plots ------------------------------------------------
step "7  gap metrics + shift evaluation"
$PY scripts/measure_gap.py --models $MODELS | tee logs/gpu/measure_gap.log
$PY scripts/shift_eval.py --models $MODELS | tee logs/gpu/shift_eval.log
$PY scripts/plot_shift.py | tee logs/gpu/plot_shift.log

# 8. Pack results --------------------------------------------------------------------
step "8  pack results"
items=(results/gap_metrics.csv results/shift_eval.csv results/finetune_runs.csv
       results/finetune_selection.json results/figures logs/gpu)
items+=($(ls runs/*/log.jsonl runs/*/config.json runs/*/train.log 2>/dev/null))
[[ "$INCLUDE_EMB" == "1" ]] && items+=(results/embeddings)
tar czf results_gpu.tar.gz "${items[@]}"
ls -lh results_gpu.tar.gz
echo "Done. Send back results_gpu.tar.gz"
