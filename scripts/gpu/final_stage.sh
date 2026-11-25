#!/usr/bin/env bash
# Final stage after encoding (steps 7-9 of run_gpu.sh), at maximum parallelism:
#   CPU : shift evaluation split into model x stat x mode x benchmark tasks on a 64-worker pool
#   GPU : seed-variance checkpoints encoded/evaluated by 4 workers (one per card),
#         naturalness on GPU 0, the last sweep runs on GPUs 1-3
# Every task is retried on failure and skipped if its output exists; no step
# aborts the script. Usage (package root): bash scripts/gpu/final_stage.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$PWD/../env/bin/python}"
CORE_MODELS="clip negclip negclip_gelu ft_plain ft_hardimg ft_hardcap ft_negclip ft_plain512"
CPU_WORKERS="${CPU_WORKERS:-64}"
export XDG_CACHE_HOME="$PWD/.cache" XDG_CONFIG_HOME="$PWD/.cache/config" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" NLTK_DATA="$PWD/.cache/nltk_data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/gpu results/shift_eval results/seed_variance .p
step() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

missing=""
for m in $CORE_MODELS; do [[ -f results/embeddings/$m/sugarcrepe.npz ]] || missing+=" $m"; done
if [[ -n "$missing" ]]; then
    step "6  encode missing core embeddings:$missing"
    "$PY" scripts/gpu/task_pool.py <("$PY" -c "
import json, sys
for m in sys.argv[1:]:
    print(json.dumps(dict(name=f'embed_{m}', out=f'results/embeddings/{m}/sugarcrepe.npz', script='scripts/embed_all.py',
        args=['--models', m, '--benchmarks', 'coco_val', 'coco_test', 'coco_order', 'vg_relation', 'vg_attribution', 'sugarcrepe', '--workers', '8'],
        env=dict(CUDA_VISIBLE_DEVICES='0'))))" $missing) --workers 2 || echo "WARNING: encoding pool error"
fi
[[ -f results/gap_metrics.csv ]] || "$PY" scripts/measure_gap.py --models $CORE_MODELS > logs/gpu/measure_gap.log 2>&1 \
    || echo "WARNING: measure_gap failed"

step "7  shift evaluation: task pool on $CPU_WORKERS CPU workers"
"$PY" - $CORE_MODELS > .p/shift_tasks.jsonl <<'EOF'
import json, sys
one = dict(OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1)
for m in sys.argv[1:]:
    for stat in ["mean", "median"]:
        for mode in ["text", "both", "text_norenorm", "orth"]:
            for b in ["coco_test", "coco_order", "vg_relation", "vg_attribution", "sugarcrepe"]:
                out = f"results/shift_eval/{m}__{stat}__{mode}__{b}.csv"
                print(json.dumps(dict(name=f"se_{m}__{stat}__{mode}__{b}", out=out, script="scripts/shift_eval.py",
                    args=["--models", m, "--benchmarks", b, "--modes", mode, "--stats", stat, "--out", out], env=one)))
# clean accuracy of the core models for the seed-variance table
print(json.dumps(dict(name="sv_core", out="results/seed_variance/_core.csv", script="scripts/shift_eval.py",
    args=["--models", *sys.argv[1:], "--benchmarks", "coco_test", "coco_order", "vg_relation", "vg_attribution",
          "--modes", "text", "--stats", "mean", "--alphas", "0", "--targets", "--sigmas", "0",
          "--out", "results/seed_variance/_core.csv"], env=one)))
EOF
"$PY" scripts/gpu/task_pool.py .p/shift_tasks.jsonl --workers "$CPU_WORKERS" > logs/gpu/shift_pool.log 2>&1 &
SHIFT_POOL=$!

step "8  naturalness (GPU 0, background)"
( CUDA_VISIBLE_DEVICES=0 "$PY" scripts/naturalness.py --models $CORE_MODELS --lm "$PWD/.cache/gpt2" \
    > logs/gpu/naturalness.log 2>&1 || echo "WARNING: naturalness failed, see logs/gpu/naturalness.log" ) &
NAT=$!

sv_round() {
    "$PY" - > .p/sv_tasks.jsonl <<'EOF'
import glob, json, os
ms = sorted(os.path.basename(p)[:-3] for p in glob.glob("checkpoints/ft_*_lr*_s*.pt"))
for i, m in enumerate(ms):
    print(json.dumps(dict(name=f"sv_{m}", out=f"results/seed_variance/{m}.csv", script="scripts/gpu/seed_var_one.py",
        args=["--model", m, "--out", f"results/seed_variance/{m}.csv"], env=dict(CUDA_VISIBLE_DEVICES=str(i % 4)))))
EOF
    "$PY" scripts/gpu/task_pool.py .p/sv_tasks.jsonl --workers 4 >> logs/gpu/sv_pool.log 2>&1 \
        || echo "WARNING: seed-variance pool error"
}
step "7b seed variance, round 1 (finished checkpoints, 4 GPU workers)"
sv_round

step "4b wait for the last sweep runs"
while pgrep -u "$USER" -f "scripts/gpu/run_[s]weep.py" > /dev/null; do sleep 60; done
"$PY" scripts/gpu/select_best.py > logs/gpu/select_best.log 2>&1 || echo "WARNING: select_best failed"
cat logs/gpu/select_best.log
step "7b seed variance, round 2 (late checkpoints)"
sv_round
"$PY" - <<'EOF'
import glob, pandas as pd
dfs = []
for f in glob.glob("results/seed_variance/*.csv"):
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print("skip", f, e)
pd.concat(dfs, ignore_index=True).to_csv("results/seed_variance.csv", index=False)
print("seed_variance.csv:", sum(len(d) for d in dfs), "rows from", len(dfs), "files")
EOF

step "7  wait for shift pool + naturalness, merge, plot"
wait $SHIFT_POOL || echo "WARNING: shift pool error"
tail -3 logs/gpu/shift_pool.log
wait $NAT
"$PY" - <<'EOF'
import glob, pandas as pd
dfs = []
for f in glob.glob("results/shift_eval/*__*.csv"):
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print("skip", f, e)
df = pd.concat(dfs, ignore_index=True)
df.to_csv("results/shift_eval.csv", index=False)
print("shift_eval.csv:", len(df), "rows from", len(dfs), "files; models:", sorted(df.model.unique()))
EOF
"$PY" scripts/plot_shift.py > logs/gpu/plot_shift.log 2>&1 || echo "WARNING: plot_shift failed, see logs/gpu/plot_shift.log"

step "9  pack results"
candidates=(results/gap_metrics.csv results/shift_eval.csv results/seed_variance.csv results/seed_variance
       results/finetune_runs.csv results/finetune_selection.json results/naturalness.csv results/figures logs/gpu)
candidates+=($(ls runs/*/log.jsonl runs/*/config.json runs/*/train.log 2>/dev/null))
items=()
for it in "${candidates[@]}"; do [[ -e "$it" ]] && items+=("$it") || echo "pack: skipping missing $it"; done
tar czf results_gpu.tar.gz "${items[@]}" && ls -lh results_gpu.tar.gz
echo "ALL DONE"
