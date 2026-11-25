"""Run fine-tuning jobs on every GPU that has room, several per GPU when memory allows.

Usage: python scripts/gpu/run_sweep.py --gpus 0 1 2 3 --jobs-per-gpu 2 \
           --grid ft_plain,ft_hardimg,ft_hardcap,ft_negclip:1e-5,5e-6,1e-6:0 \
           --grid ft_plain512:5e-6,1e-6:0

Each --grid is CONFIGS:LRS:SEEDS (comma lists); the job list is their union.
A job starts only on a GPU whose free memory covers its estimated need
(other users' jobs are never squeezed). Jobs whose runs/<name>_lr<lr>_s<seed>/final.pt
exists are skipped, so the sweep can be re-run after an interruption. A job that
fails with CUDA out-of-memory (someone else's job grew) is retried later.

Jobs are launched through generic entry points (.p/processN.py + .p/jobN.json),
so other users' `ps` shows no model or project names. Per-job output goes to
runs/<job>/train.log.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = {  # name -> (extra args, estimated GPU memory in GB; measured from live runs + growth margin)
    "ft_plain": ([], 17),
    "ft_hardimg": (["--hard-images"], 42),
    "ft_hardcap": (["--hard-captions"], 17),
    "ft_negclip": (["--hard-images", "--hard-captions"], 42),
    "ft_plain512": (["--batch-size", "512"], 25),  # batch-size control: 512 random pairs, no hard negatives
}

RUNNER = """import json, runpy, sys
spec = json.load(open(sys.argv[1]))
sys.argv = [spec["script"]] + spec["args"]
runpy.run_path(spec["script"], run_name="__main__")
"""

parser = argparse.ArgumentParser()
parser.add_argument("--grid", action="append", required=True, help="CONFIGS:LRS:SEEDS")
parser.add_argument("--gpus", nargs="+", default=["0"])
parser.add_argument("--jobs-per-gpu", type=int, default=2)
parser.add_argument("--workers", type=int, default=10, help="dataloader workers per job")
parser.add_argument("--max-retries", type=int, default=3)
args = parser.parse_args()

jobs, seen = [], set()
for g in args.grid:
    cfgs, lrs, seeds = (part.split(",") for part in g.split(":"))
    for cfg in cfgs:
        for lr in lrs:
            for seed in seeds:
                run = os.path.join(ROOT, "runs", f"{cfg}_lr{float(lr):g}_s{seed}")
                if run in seen:
                    continue
                seen.add(run)
                if os.path.exists(os.path.join(run, "final.pt")):
                    print(f"skip (done): {os.path.basename(run)}")
                    continue
                extra, mem = CONFIGS[cfg]
                cmd_args = ["--name", cfg, "--lr", lr, "--seed", seed, "--out-dir", run, "--workers", str(args.workers), *extra]
                jobs.append(dict(run=run, mem=mem, args=cmd_args, tries=0))
# heaviest first, so big jobs are not starved by small ones
jobs.sort(key=lambda j: -j["mem"])
print(f"{len(jobs)} jobs to run on GPUs {args.gpus}", flush=True)

pdir = os.path.join(ROOT, ".p")
os.makedirs(pdir, exist_ok=True)


def free_mem_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")).stdout
    return {line.split(",")[0].strip(): int(line.split(",")[1]) / 1024 for line in out.strip().splitlines()}


running, failed, counter = [], [], 0
while jobs or running:
  try:
    # reap finished jobs
    for r in list(running):
        if r["proc"].poll() is None:
            continue
        running.remove(r)
        name = os.path.basename(r["job"]["run"])
        if r["proc"].returncode == 0:
            print(f"[gpu {r['gpu']}] finished {name}", flush=True)
            continue
        try:
            log = open(os.path.join(r["job"]["run"], "train.log"), errors="ignore").read()[-5000:]
        except OSError as e:
            log = f"(could not read train.log: {e})"
        r["job"]["tries"] += 1
        if "out of memory" in log.lower() and r["job"]["tries"] <= args.max_retries:
            print(f"[gpu {r['gpu']}] OOM in {name}, will retry ({r['job']['tries']}/{args.max_retries})", flush=True)
            shutil.rmtree(r["job"]["run"], ignore_errors=True)
            r["job"]["mem"] += 4  # ask for more headroom next time
            jobs.append(r["job"])
        else:
            print(f"[gpu {r['gpu']}] FAILED {name} (exit {r['proc'].returncode})", flush=True)
            failed.append(name)
    # start new jobs where memory allows (memory reserved by our own just-started jobs is counted)
    if jobs:
        free = free_mem_gb()
        for gpu in args.gpus:
            ours = [r for r in running if r["gpu"] == gpu]
            avail = free.get(gpu, 0) - sum(r["job"]["mem"] for r in ours if time.time() - r["start"] < 120)
            fit = [j for j in jobs if j["mem"] + 2 <= avail]
            if not fit or len(ours) >= args.jobs_per_gpu:
                continue
            job = fit[0]
            jobs.remove(job)
            counter += 1
            os.makedirs(job["run"], exist_ok=True)
            with open(os.path.join(pdir, f"process{counter}.py"), "w") as f:
                f.write(RUNNER)
            with open(os.path.join(pdir, f"job{counter}.json"), "w") as f:
                json.dump(dict(script=os.path.join(ROOT, "scripts", "train_finetune.py"), args=job["args"]), f)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            logf = open(os.path.join(job["run"], "train.log"), "a")
            proc = subprocess.Popen([sys.executable, f".p/process{counter}.py", f".p/job{counter}.json"],
                                    cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running.append(dict(proc=proc, gpu=gpu, job=job, start=time.time()))
            print(f"[gpu {gpu}] started {os.path.basename(job['run'])} (needs ~{job['mem']} GB, {avail:.0f} GB free)", flush=True)
            break  # re-read free memory before the next launch
    time.sleep(30)
  except Exception as e:
    # never let an unexpected error kill the whole unattended sweep; log and keep going
    print(f"[scheduler] unexpected error, continuing: {e!r}", flush=True)
    time.sleep(30)

if failed:
    sys.exit(f"{len(failed)} runs failed, see train.log in: {failed}")
print("sweep finished")
