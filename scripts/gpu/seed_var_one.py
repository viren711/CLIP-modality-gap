"""Clean (alpha=0, no noise) accuracy of one extra checkpoint for the seed-variance table:
encode its benchmarks, evaluate, then delete the embeddings to keep disk use small.

Usage: python scripts/gpu/seed_var_one.py --model ft_negclip_lr1e-06_s1 --out results/seed_variance/X.csv
Runs embed_all.py and shift_eval.py in-process (no model names in `ps`).
"""
import argparse
import os
import runpy
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = ["coco_test", "coco_order", "vg_relation", "vg_attribution"]

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--workers", default="6")
a = p.parse_args()


def call(script, args):
    sys.argv = [script] + args
    runpy.run_path(os.path.join(ROOT, script), run_name="__main__")


try:
    # coco_val too: shift_eval.py always fits the gap vector on it, even at alpha = 0
    call("scripts/embed_all.py", ["--models", a.model, "--benchmarks", "coco_val", *BENCH, "--workers", a.workers])
    call("scripts/shift_eval.py", ["--models", a.model, "--benchmarks", *BENCH, "--modes", "text", "--stats", "mean",
                                   "--alphas", "0", "--targets", "--sigmas", "0", "--out", a.out])
finally:
    shutil.rmtree(os.path.join(ROOT, "results", "embeddings", a.model), ignore_errors=True)
