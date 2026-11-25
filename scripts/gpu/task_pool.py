"""Run many independent python-script tasks in parallel, robustly.

Usage: python scripts/gpu/task_pool.py TASKS.jsonl --workers 64 [--retries 2] [--timeout 7200]
Each line of TASKS.jsonl: {"name": ..., "out": path, "script": path, "args": [...], "env": {...}}
A task whose `out` already exists is skipped, so the pool can be re-run after an
interruption. A failed or timed-out task is retried; a task that keeps failing is
reported but never stops the others. Tasks run through generic entry points
(.p/runner.py + .p/tN.json), so other users' `ps` shows no model or project names.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RUNNER = """import json, runpy, sys
spec = json.load(open(sys.argv[1]))
sys.argv = [spec["script"]] + spec["args"]
runpy.run_path(spec["script"], run_name="__main__")
"""

p = argparse.ArgumentParser()
p.add_argument("tasks")
p.add_argument("--workers", type=int, default=64)
p.add_argument("--retries", type=int, default=2)
p.add_argument("--timeout", type=float, default=7200, help="seconds per attempt")
p.add_argument("--log-dir", default="logs/gpu/tasks")
a = p.parse_args()

os.makedirs(a.log_dir, exist_ok=True)
os.makedirs(".p", exist_ok=True)
with open(".p/runner.py", "w") as f:
    f.write(RUNNER)
tag = os.path.splitext(os.path.basename(a.tasks))[0]
tasks = [json.loads(line) for line in open(a.tasks) if line.strip()]
todo = [t for t in tasks if not os.path.exists(t["out"])]
print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run on {a.workers} workers", flush=True)


def run(i, t):
    spec = f".p/{tag}_{i}.json"
    with open(spec, "w") as f:
        json.dump(dict(script=t["script"], args=t["args"]), f)
    log = os.path.join(a.log_dir, t["name"] + ".log")
    env = dict(os.environ, **{k: str(v) for k, v in t.get("env", {}).items()})
    for attempt in range(1, a.retries + 2):
        try:
            with open(log, "a") as f:
                r = subprocess.run([sys.executable, ".p/runner.py", spec], stdout=f, stderr=subprocess.STDOUT,
                                   env=env, timeout=a.timeout)
            if r.returncode == 0 and os.path.exists(t["out"]):
                return t["name"], True, attempt
            msg = f"exit {r.returncode}" + ("" if os.path.exists(t["out"]) else ", no output")
        except subprocess.TimeoutExpired:
            msg = f"timeout after {a.timeout:.0f}s"
        except Exception as e:  # never let one task take down the pool
            msg = repr(e)
        with open(log, "a") as f:
            f.write(f"\n[pool] attempt {attempt} failed: {msg}\n")
        time.sleep(10 * attempt)
    return t["name"], False, attempt


failed, n_done, t0 = [], 0, time.time()
with ThreadPoolExecutor(a.workers) as ex:
    futs = [ex.submit(run, i, t) for i, t in enumerate(todo)]
    for fut in as_completed(futs):
        try:
            name, ok, _ = fut.result()
        except Exception as e:
            name, ok = f"? ({e!r})", False
        n_done += 1
        if not ok:
            failed.append(name)
        if not ok or n_done % 10 == 0 or n_done == len(todo):
            el = time.time() - t0
            eta = el / n_done * (len(todo) - n_done)
            print(f"[pool {el / 60:5.1f} min] {n_done}/{len(todo)} done, eta {eta / 60:.0f} min"
                  + (f"  FAILED: {name}" if not ok else ""), flush=True)
print(f"pool finished: {len(todo) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
