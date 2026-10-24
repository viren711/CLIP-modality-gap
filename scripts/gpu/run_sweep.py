"""Run the fine-tuning grid: 4 configs x learning rates x seeds, one job per GPU.

Usage: python scripts/gpu/run_sweep.py --gpus 0 1 --lrs 1e-5 5e-6 1e-6 --seeds 0
Jobs whose runs/<name>_lr<lr>_s<seed>/final.pt already exists are skipped,
so the sweep can be re-run after an interruption. Per-job stdout goes to
runs/<job>/train.log.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = {
    "ft_plain": [],
    "ft_hardimg": ["--hard-images"],
    "ft_hardcap": ["--hard-captions"],
    "ft_negclip": ["--hard-images", "--hard-captions"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
parser.add_argument("--lrs", nargs="+", default=["1e-5", "5e-6", "1e-6"])
parser.add_argument("--seeds", nargs="+", default=["0"])
parser.add_argument("--gpus", nargs="+", default=["0"])
parser.add_argument("--extra", default="", help="extra args for train_finetune.py, e.g. '--workers 8'")
args = parser.parse_args()

jobs = []
for cfg, lr, seed in itertools.product(args.configs, args.lrs, args.seeds):
    run = os.path.join(ROOT, "runs", f"{cfg}_lr{float(lr):g}_s{seed}")
    if os.path.exists(os.path.join(run, "final.pt")):
        print(f"skip (done): {run}")
        continue
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_finetune.py"), "--name", cfg, "--lr", lr,
           "--seed", seed, "--out-dir", run, *CONFIGS[cfg], *args.extra.split()]
    jobs.append((run, cmd))

running = {}  # gpu -> (run, process)
failed = []
while jobs or running:
    for gpu in args.gpus:
        if gpu in running:
            run, proc = running[gpu]
            if proc.poll() is None:
                continue
            print(f"[gpu {gpu}] finished {os.path.basename(run)} (exit {proc.returncode})", flush=True)
            if proc.returncode != 0:
                failed.append(run)
            del running[gpu]
        if jobs:
            run, cmd = jobs.pop(0)
            os.makedirs(run, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            logf = open(os.path.join(run, "train.log"), "a")
            running[gpu] = (run, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT))
            print(f"[gpu {gpu}] started {os.path.basename(run)}", flush=True)
    time.sleep(10)

if failed:
    sys.exit(f"{len(failed)} runs failed, see train.log in: {failed}")
print("sweep finished")
