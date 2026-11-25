#!/usr/bin/env bash
# Resume run_gpu.sh after an interrupted step 4: finish the remaining sweep runs
# on GPUs 1-3 while steps 5-9 run on GPU 0 / CPU, using only finished runs for
# the core models (best LR is picked on seed 0, which is complete) and adding
# the late seed runs to the seed-variance table when they finish.
# Usage (from the package root): bash scripts/gpu/post_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$PWD/../env/bin/python}"
CORE_MODELS="clip negclip negclip_gelu ft_plain ft_hardimg ft_hardcap ft_negclip ft_plain512"
SV_BENCH="coco_test coco_order vg_relation vg_attribution"
GPT2_DIR="$PWD/.cache/gpt2"
mkdir -p logs/gpu results/shift_eval results/seed_variance .p

export XDG_CACHE_HOME="$PWD/.cache" XDG_CONFIG_HOME="$PWD/.cache/config" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" PIP_CACHE_DIR="$PWD/.cache/pip"
export NLTK_DATA="$PWD/.cache/nltk_data" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# make CUDA device numbers match nvidia-smi, so "GPU 0" is the card kept free of training
export CUDA_DEVICE_ORDER=PCI_BUS_ID
step() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

# generic entry points; the id comes from the (sub)shell pid, not a counter, because
# backgrounded calls run in subshells that would all reuse the same counter value
anon() {
    local n="x${BASHPID}_${RANDOM}"
    printf 'import json, runpy, sys\nspec = json.load(open(sys.argv[1]))\nsys.argv = [spec["script"]] + spec["args"]\nrunpy.run_path(spec["script"], run_name="__main__")\n' > ".p/process${n}.py"
    python3 -c "import json,sys; json.dump({'script': sys.argv[1], 'args': sys.argv[2:]}, open('.p/job${n}.json','w'))" "$@"
    "$PY" ".p/process${n}.py" ".p/job${n}.json"
}

extra_models() { ls checkpoints/ft_*_lr*_s*.pt 2>/dev/null | xargs -rn1 basename | sed 's/\.pt$//'; }

# encode -> clean eval -> delete embeddings, for every extra checkpoint not done yet
seed_variance_worker() {  # $1 = worker index, $2 = number of workers
    local i=0
    for m in $(extra_models); do
        i=$((i + 1))
        (( (i - 1) % $2 == $1 )) || continue
        [[ -f results/seed_variance/${m}.csv ]] && continue
        CUDA_VISIBLE_DEVICES=0 anon scripts/embed_all.py --models "$m" --benchmarks $SV_BENCH --workers 8 2>&1 \
            | grep -v "it/s]" >> logs/gpu/embed_sv.log
        anon scripts/shift_eval.py --models "$m" --benchmarks $SV_BENCH \
            --modes text --stats mean --alphas 0 --targets --sigmas 0 \
            --out "results/seed_variance/${m}.csv" >> logs/gpu/seed_variance.log 2>&1 \
            || echo "WARNING: seed-variance eval failed for $m"
        rm -rf "results/embeddings/$m"
    done
}

step "4  remaining sweep runs on GPUs 1-3 (one heavy job per card)"
if pgrep -u "$USER" -f "scripts/gpu/run_sweep.py" > /dev/null; then
    echo "sweep already running, attaching to it"
else
"$PY" scripts/gpu/run_sweep.py --gpus 1 2 3 --jobs-per-gpu 1 \
    --grid "ft_plain,ft_hardimg,ft_hardcap,ft_negclip:1e-5,5e-6,1e-6:0" \
    --grid "ft_plain,ft_hardimg,ft_hardcap,ft_negclip:5e-6,1e-6:1,2" \
    --grid "ft_plain512:5e-6,1e-6:0" >> logs/gpu/sweep.log 2>&1 &
fi

step "5  select best learning rate per config (seed 0 runs are complete)"
anon scripts/gpu/select_best.py | tee logs/gpu/select_best.log

step "6  encode benchmarks: core models (2 encoders on GPU 0)"
CUDA_VISIBLE_DEVICES=0 anon scripts/embed_all.py --models clip negclip negclip_gelu ft_plain \
    --benchmarks coco_val coco_test coco_order vg_relation vg_attribution sugarcrepe --workers 12 2>&1 \
    | grep -v "it/s]" > logs/gpu/embed_a.log &
E1=$!
CUDA_VISIBLE_DEVICES=0 anon scripts/embed_all.py --models ft_hardimg ft_hardcap ft_negclip ft_plain512 \
    --benchmarks coco_val coco_test coco_order vg_relation vg_attribution sugarcrepe --workers 12 2>&1 \
    | grep -v "it/s]" > logs/gpu/embed_b.log &
E2=$!
wait $E1 $E2
cat logs/gpu/embed_a.log logs/gpu/embed_b.log > logs/gpu/embed.log

step "7  gap metrics + shift evaluation (parallel, CPU) | 8 naturalness + 7b seed variance (GPU 0)"
anon scripts/measure_gap.py --models $CORE_MODELS | tee logs/gpu/measure_gap.log
THREADS=$(( $(nproc) / 8 ))
pids=()
for m in $CORE_MODELS; do
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS anon scripts/shift_eval.py --models "$m" \
        --benchmarks coco_test coco_order vg_relation vg_attribution sugarcrepe \
        --modes text both text_norenorm orth \
        --out "results/shift_eval/${m}.csv" > "logs/gpu/shift_eval_${m}.log" 2>&1 &
    pids+=($!)
done

CUDA_VISIBLE_DEVICES=0 anon scripts/naturalness.py --models $CORE_MODELS --lm "$GPT2_DIR" \
    > logs/gpu/naturalness.log 2>&1 || echo "WARNING: naturalness failed, see logs/gpu/naturalness.log"

anon scripts/shift_eval.py --models $CORE_MODELS --benchmarks $SV_BENCH \
    --modes text --stats mean --alphas 0 --targets --sigmas 0 \
    --out results/seed_variance/_core.csv > logs/gpu/seed_variance.log 2>&1
seed_variance_worker 0 2 &
W1=$!
seed_variance_worker 1 2 &
W2=$!
wait $W1 $W2

step "4b wait for the last sweep runs, export them, add to seed variance"
while pgrep -u "$USER" -f "scripts/gpu/run_sweep.py" > /dev/null; do sleep 60; done
grep -q "runs failed" logs/gpu/sweep.log && echo "WARNING: sweep reported failed runs, see logs/gpu/sweep.log"
anon scripts/gpu/select_best.py | tee logs/gpu/select_best.log
seed_variance_worker 0 1
"$PY" -c "
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in glob.glob('results/seed_variance/*.csv')], ignore_index=True).to_csv('results/seed_variance.csv', index=False)
print('seed_variance.csv rows:', sum(1 for _ in open('results/seed_variance.csv')) - 1)
"

step "7  wait for shift evaluation, merge, plot"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
cat logs/gpu/shift_eval_*.log > logs/gpu/shift_eval.log
[[ $fail -eq 0 ]] || echo "WARNING: one or more shift_eval.py workers failed, see logs/gpu/shift_eval_*.log"
"$PY" -c "
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in glob.glob('results/shift_eval/*.csv')], ignore_index=True).to_csv('results/shift_eval.csv', index=False)
"
anon scripts/plot_shift.py | tee logs/gpu/plot_shift.log

step "9  pack results"
candidates=(results/gap_metrics.csv results/shift_eval.csv results/shift_eval results/seed_variance.csv
       results/finetune_runs.csv results/finetune_selection.json
       results/naturalness.csv results/figures logs/gpu)
candidates+=($(ls runs/*/log.jsonl runs/*/config.json runs/*/train.log 2>/dev/null))
items=()
for it in "${candidates[@]}"; do [[ -e "$it" ]] && items+=("$it") || echo "pack: skipping missing $it"; done
tar czf results_gpu.tar.gz "${items[@]}"
ls -lh results_gpu.tar.gz
echo "ALL DONE"
