#!/usr/bin/env bash
# =========================================================================
#  CS 725 - modality gap x hard negatives: everything that needs the GPU.
#
#  Run from the unzipped package root:
#      bash run_gpu.sh
#  Options (environment variables):
#      GPUS="0 1 2 3"     GPUs to use                                    (default "0")
#      JOBS_PER_GPU=2     training jobs started per GPU when memory allows (default 2)
#      LRS="1e-5 5e-6 1e-6"  learning-rate sweep for the 4 core configs
#      SEEDS="0"          seeds per (config, lr) for the core sweep; e.g. "0 1 2"
#      EXTRA_SEEDS="1 2"  extra seeds, at the two best LRs only, for error bars
#      BATCH512_LRS="5e-6 1e-6"  LRs for the ft_plain512 batch-size control
#      PY=/path/python    use an existing Python that has CUDA torch
#                         (otherwise a venv .venv-gpu is created)
#      INCLUDE_EMB=1      put cached embeddings (~3 GB) in the results tarball
#      TORCH_INDEX=url    torch wheel index, e.g. https://download.pytorch.org/whl/cu126
#  Every checkpoint is kept (checkpoints/<config>_lr<lr>_s<seed>.pt, hard
#  links to runs/.../final.pt - no extra disk), plus the best per config as
#  checkpoints/<config>.pt. All caches (pip, model weights, nltk, matplotlib)
#  stay inside this folder. Every step is resumable.
#
#  Steps
#    0  Python env                 5  pick best LR per config -> checkpoints/
#    1  GPU check + smoke test     6  encode all benchmarks with all models
#    2  full COCO 2014 images      7  gap metrics, shift sweep (+ H3 controls), plots
#    2b SugarCrepe + val2017       8  caption-naturalness (GPT-2 perplexity) analysis
#    3  leakage check              9  pack results -> results_gpu.tar.gz
#    4  fine-tuning sweep (queued across all GPUs, retried on OOM)
#  Send back: results_gpu.tar.gz
# =========================================================================
set -euo pipefail
cd "$(dirname "$0")"
GPUS="${GPUS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
LRS="${LRS:-1e-5,5e-6,1e-6}"
SEEDS="${SEEDS:-0}"
EXTRA_SEEDS="${EXTRA_SEEDS:-1,2}"
BATCH512_LRS="${BATCH512_LRS:-5e-6,1e-6}"
INCLUDE_EMB="${INCLUDE_EMB:-0}"
CORE_MODELS="clip negclip negclip_gelu ft_plain ft_hardimg ft_hardcap ft_negclip ft_plain512"
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

# anon SCRIPT ARGS... : run a python script through a generic process<N>.py
# entry point reading its real command from job<N>.json, so another user's
# `ps`/`nvidia-smi` on this shared box shows no model, dataset or course names.
mkdir -p .p
ANON_N=0
anon() {
    ANON_N=$((ANON_N + 1))
    cat > ".p/process${ANON_N}.py" <<'PYEOF'
import json, runpy, sys
spec = json.load(open(sys.argv[1]))
sys.argv = [spec["script"]] + spec["args"]
runpy.run_path(spec["script"], run_name="__main__")
PYEOF
    python3 -c "import json,sys; json.dump({'script': sys.argv[1], 'args': sys.argv[2:]}, open('.p/job${ANON_N}.json','w'))" "$@"
    "$PY" ".p/process${ANON_N}.py" ".p/job${ANON_N}.json"
}

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
# only hit the network if punkt_tab is not already cached (the download call can
# hang on a restricted-network server even when the data is present)
[[ -d "$NLTK_DATA/tokenizers/punkt_tab" ]] || \
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

# 2b. SugarCrepe annotations + COCO val2017 images they reference -------------
# (download_data.sh is idempotent: COCO annotations/Karpathy/ARO/NegCLIP steps
# are already satisfied from the package and are skipped; only SugarCrepe runs)
step "2b  SugarCrepe data"
PY="$PY" bash scripts/download_data.sh 2>&1 | tee -a logs/gpu/prepare_coco.log

if [[ ! -f runs/_smoke_ok ]]; then
    step "1b smoke test: 20 steps of the heaviest config (NegCLIP, batch 256)"
    CUDA_VISIBLE_DEVICES="${GPUS%% *}" anon scripts/train_finetune.py --name _smoke --hard-images --hard-captions \
        --epochs 1 --max-steps 20 --out-dir runs/_smoke 2>&1 | tail -5
    rm -rf runs/_smoke && touch runs/_smoke_ok
fi

# 3. Leakage check --------------------------------------------------------------
step "3  leakage check"
anon scripts/check_overlap.py | tee logs/gpu/check_overlap.log

# 4. Fine-tuning sweep: core grid + extra seeds + batch-512 control, queued --
# across all GPUs (2 jobs/GPU by default, auto-skips GPUs that don't have room,
# retries on OOM). Entry points are anonymised (.p/processN.py + jobN.json),
# so another user's `ps` shows no model or project names.
step "4  fine-tuning sweep: GPUs [$GPUS], up to $JOBS_PER_GPU jobs/GPU"
# don't let a handful of permanently-failed runs (after OOM retries exhausted)
# kill the rest of the unattended pipeline: every other step works off whatever
# checkpoints exist, so run with pipefail off just for this one step.
set +e +o pipefail
$PY scripts/gpu/run_sweep.py --gpus $GPUS --jobs-per-gpu "$JOBS_PER_GPU" \
    --grid "ft_plain,ft_hardimg,ft_hardcap,ft_negclip:${LRS}:${SEEDS}" \
    --grid "ft_plain,ft_hardimg,ft_hardcap,ft_negclip:${BATCH512_LRS}:${EXTRA_SEEDS}" \
    --grid "ft_plain512:${BATCH512_LRS}:${SEEDS}" \
    2>&1 | tee -a logs/gpu/sweep.log
[[ ${PIPESTATUS[0]} -eq 0 ]] || echo "WARNING: sweep reported failed runs, see logs/gpu/sweep.log - continuing with whatever succeeded"
set -e -o pipefail

# 5. Select best LR per config -> checkpoints/<config>.pt, keep every run -----
step "5  select best learning rate per config (every run kept)"
anon scripts/gpu/select_best.py | tee logs/gpu/select_best.log
ALL_MODELS=$(ls checkpoints/*.pt 2>/dev/null | xargs -n1 basename | sed 's/\.pt$//')

# 6. Encode all benchmarks with all models (core models on GPU; the many extra
#    seed/LR checkpoints only need the clean benchmarks for the seed-variance
#    table, no full shift grid) -----------------------------------------------
step "6  encode benchmarks: core models"
CUDA_VISIBLE_DEVICES="${GPUS%% *}" anon scripts/embed_all.py --models $CORE_MODELS \
    --benchmarks coco_val coco_test coco_order vg_relation vg_attribution sugarcrepe --workers 8 2>&1 \
    | grep -v "it/s]" | tee logs/gpu/embed.log
# (the many extra seed/LR checkpoints are encoded one at a time in step 7b, then
#  their embeddings are deleted, to stay inside the shared /tmp disk budget)

# 7. Gap metrics + shift sweep (incl. H3 controls), one process per model ------
# on the 96-core machine so the ~1.5 h/model CPU grid runs in parallel instead
# of in series (7+ h) -----------------------------------------------------------
step "7  gap metrics + shift evaluation (parallel across $(echo $CORE_MODELS | wc -w) models)"
anon scripts/measure_gap.py --models $CORE_MODELS | tee logs/gpu/measure_gap.log
mkdir -p results/shift_eval
N_MODELS=$(echo $CORE_MODELS | wc -w)
THREADS=$(( $(nproc) / N_MODELS > 0 ? $(nproc) / N_MODELS : 1 ))
pids=()
for m in $CORE_MODELS; do
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS anon scripts/shift_eval.py --models "$m" \
        --benchmarks coco_test coco_order vg_relation vg_attribution sugarcrepe \
        --modes text both text_norenorm orth \
        --out "results/shift_eval/${m}.csv" > "logs/gpu/shift_eval_${m}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
cat logs/gpu/shift_eval_*.log | tee logs/gpu/shift_eval.log > /dev/null
[[ $fail -eq 0 ]] || echo "WARNING: one or more shift_eval.py workers failed, see logs/gpu/shift_eval_*.log - continuing"
$PY -c "
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in glob.glob('results/shift_eval/*.csv')], ignore_index=True).to_csv('results/shift_eval.csv', index=False)
"
anon scripts/plot_shift.py | tee logs/gpu/plot_shift.log

# 7b. Seed-variance + matched-LR table: clean accuracy (alpha=0, no noise) over
#     every trained checkpoint. Disk-safe on the shared /tmp: core models reuse
#     their step-6 embeddings; every other checkpoint is encoded, evaluated, and
#     its embeddings deleted before the next, so at most ~0.6 GB of extra
#     embeddings sits on disk at once.
step "7b  seed-variance + matched-LR table (clean accuracy, all checkpoints)"
mkdir -p results/seed_variance
SV_BENCH="coco_test coco_order vg_relation vg_attribution"
anon scripts/shift_eval.py --models $CORE_MODELS --benchmarks $SV_BENCH \
    --modes text --stats mean --alphas 0 --targets --sigmas 0 \
    --out results/seed_variance/_core.csv | tee logs/gpu/seed_variance.log
for m in $ALL_MODELS; do
    case " $CORE_MODELS " in *" $m "*) continue ;; esac
    [[ -f results/seed_variance/${m}.csv ]] && continue
    CUDA_VISIBLE_DEVICES="${GPUS%% *}" anon scripts/embed_all.py --models "$m" \
        --benchmarks $SV_BENCH --workers 8 2>&1 | grep -v "it/s]" >> logs/gpu/embed.log
    anon scripts/shift_eval.py --models "$m" --benchmarks $SV_BENCH \
        --modes text --stats mean --alphas 0 --targets --sigmas 0 \
        --out "results/seed_variance/${m}.csv" >> logs/gpu/seed_variance.log 2>&1
    rm -rf "results/embeddings/$m"   # free disk; keep only the small csv
done
$PY -c "
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in glob.glob('results/seed_variance/*.csv')], ignore_index=True).to_csv('results/seed_variance.csv', index=False)
print('seed_variance.csv rows:', sum(1 for _ in open('results/seed_variance.csv')) - 1)
"

# 8. Caption-naturalness analysis (GPT-2 perplexity vs gap-direction projection)
step "8  naturalness analysis"
CUDA_VISIBLE_DEVICES="${GPUS%% *}" anon scripts/naturalness.py --models $CORE_MODELS | tee logs/gpu/naturalness.log

# 9. Pack results --------------------------------------------------------------------
step "9  pack results"
candidates=(results/gap_metrics.csv results/shift_eval.csv results/seed_variance.csv
       results/finetune_runs.csv results/finetune_selection.json
       results/naturalness.csv results/figures logs/gpu)
candidates+=($(ls runs/*/log.jsonl runs/*/config.json runs/*/train.log 2>/dev/null))
[[ "$INCLUDE_EMB" == "1" ]] && candidates+=(results/embeddings)
# pack whatever exists; a missing file (e.g. a failed step) should not prevent
# packing everything that did succeed
items=()
for it in "${candidates[@]}"; do [[ -e "$it" ]] && items+=("$it") || echo "pack: skipping missing $it"; done
tar czf results_gpu.tar.gz "${items[@]}"
ls -lh results_gpu.tar.gz
echo "Done. Send back results_gpu.tar.gz"
