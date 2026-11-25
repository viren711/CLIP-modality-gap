#!/usr/bin/env bash
# Final-report additions, run on the GPU server (package root):
#   X1  close the gap during training: 4 runs with --gap-reg, one per GPU, then
#       encode + clean/noisy text-shift evaluation of the new checkpoints
#   X2-X5  extra analyses on cached embeddings (CPU pool, starts immediately)
# Every step is a task-pool task (retries, skip-if-done, anonymised process names).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$PWD/../env/bin/python}"
export XDG_CACHE_HOME="$PWD/.cache" XDG_CONFIG_HOME="$PWD/.cache/config" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" NLTK_DATA="$PWD/.cache/nltk_data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/gpu results/extra .p
CORE="clip negclip negclip_gelu ft_plain ft_plain512 ft_hardcap ft_hardimg ft_negclip"
GR="ft_hardcap_gr1:--hard-captions:1 ft_hardcap_gr10:--hard-captions:10 ft_plain_gr10::10 ft_negclip_gr10:--hard-images,--hard-captions:10"
step() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

step "X2-X5 on cached embeddings (CPU pool, background)"
"$PY" - $CORE > .p/x_tasks.jsonl <<'EOF'
import json, sys
one = dict(OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4)
for m in sys.argv[1:]:
    out = f"results/extra/{m}.json"
    print(json.dumps(dict(name=f"x_{m}", out=out, script="scripts/extra_experiments.py", args=["--model", m, "--out", out], env=one)))
EOF
"$PY" scripts/gpu/task_pool.py .p/x_tasks.jsonl --workers 8 --retries 1 > logs/gpu/x_pool.log 2>&1 &
XPOOL=$!

step "X1 training with the gap regulariser (4 GPUs)"
"$PY" - $GR > .p/gr_tasks.jsonl <<'EOF'
import json, sys
for i, spec in enumerate(sys.argv[1:]):
    name, flags, lam = spec.split(":")
    run = f"runs/{name}_lr5e-06_s0"
    args = ["--name", name, "--lr", "5e-6", "--seed", "0", "--out-dir", run, "--workers", "16", "--gap-reg", lam]
    args += [f for f in flags.split(",") if f]
    print(json.dumps(dict(name=f"gr_{name}", out=f"{run}/final.pt", script="scripts/train_finetune.py", args=args,
                          env=dict(CUDA_VISIBLE_DEVICES=str(i), OMP_NUM_THREADS=12, MKL_NUM_THREADS=12))))
EOF
"$PY" scripts/gpu/task_pool.py .p/gr_tasks.jsonl --workers 4 --retries 1 --timeout 5400 > logs/gpu/gr_pool.log 2>&1
tail -2 logs/gpu/gr_pool.log
for spec in $GR; do
    name=${spec%%:*}
    [[ -f runs/${name}_lr5e-06_s0/final.pt ]] && ln -f runs/${name}_lr5e-06_s0/final.pt checkpoints/$name.pt
done

step "X1 encode new checkpoints (4 GPUs)"
NEW=$(for spec in $GR; do n=${spec%%:*}; [[ -f checkpoints/$n.pt ]] && echo $n; done)
"$PY" - $NEW > .p/gre_tasks.jsonl <<'EOF'
import json, sys
for i, m in enumerate(sys.argv[1:]):
    print(json.dumps(dict(name=f"gre_{m}", out=f"results/embeddings/{m}/sugarcrepe.npz", script="scripts/embed_all.py",
        args=["--models", m, "--benchmarks", "coco_val", "coco_test", "coco_order", "vg_relation", "vg_attribution",
              "sugarcrepe", "--workers", "12"], env=dict(CUDA_VISIBLE_DEVICES=str(i), OMP_NUM_THREADS=12, MKL_NUM_THREADS=12))))
EOF
"$PY" scripts/gpu/task_pool.py .p/gre_tasks.jsonl --workers 4 --retries 1 > logs/gpu/gre_pool.log 2>&1
tail -2 logs/gpu/gre_pool.log

step "X1 evaluate new checkpoints (CPU pool)"
"$PY" - $NEW > .p/grs_tasks.jsonl <<'EOF'
import json, sys
one = dict(OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2)
for m in sys.argv[1:]:
    for b in ["coco_test", "coco_order", "vg_relation", "vg_attribution", "sugarcrepe"]:
        out = f"results/extra/shift_{m}__{b}.csv"
        print(json.dumps(dict(name=f"grs_{m}_{b}", out=out, script="scripts/shift_eval.py",
            args=["--models", m, "--benchmarks", b, "--modes", "text", "--stats", "mean", "--sigmas", "0", "0.5",
                  "--out", out], env=one)))
    out = f"results/extra/{m}.json"
    print(json.dumps(dict(name=f"x_{m}", out=out, script="scripts/extra_experiments.py",
                          args=["--model", m, "--out", out], env=dict(OMP_NUM_THREADS=4))))
EOF
"$PY" scripts/gpu/task_pool.py .p/grs_tasks.jsonl --workers 32 --retries 1 > logs/gpu/grs_pool.log 2>&1
tail -2 logs/gpu/grs_pool.log

wait $XPOOL
tail -2 logs/gpu/x_pool.log
step "pack"
tar czf results_extra.tar.gz results/extra runs/*_gr*/log.jsonl runs/*_gr*/config.json logs/gpu/x_pool.log \
    logs/gpu/gr_pool.log logs/gpu/gre_pool.log logs/gpu/grs_pool.log 2>/dev/null; ls -lh results_extra.tar.gz
echo "EXTRA DONE"
