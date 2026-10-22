"""Approach 02 + 04: shift the gap post hoc and evaluate every setting.

The gap vector Delta is fitted on Karpathy *val* (held out) and applied to
every evaluation benchmark. Two families of settings:
  * alpha sweep  : fixed list of alpha values (negative = widen the gap)
  * target gap   : alpha solved per model so the remaining gap on COCO val
                   equals a target margin (so models compare at equal gap)
Each setting is evaluated clean and under Gaussian embedding noise.

Usage:
    .venv/bin/python scripts/shift_eval.py --models clip negclip
Writes results/shift_eval.csv (one row per model x setting x benchmark x sigma).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import RESULTS  # noqa: E402
from src.data import BENCHMARKS  # noqa: E402
from src.embed import get_embeddings  # noqa: E402
from src.evaluate import add_noise, eval_cases, eval_retrieval  # noqa: E402
from src.gap import alpha_for_target_gap, apply_shift, gap_vector, remaining_gap  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["clip", "negclip"])
parser.add_argument("--benchmarks", nargs="+", default=["coco_test", "coco_order", "vg_relation", "vg_attribution"])
parser.add_argument("--modes", nargs="+", default=["text", "both"])
parser.add_argument("--stats", nargs="+", default=["mean", "median"])
parser.add_argument("--alphas", nargs="+", type=float, default=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
parser.add_argument("--targets", nargs="+", type=float, default=[0.6, 0.4, 0.2, 0.1])
parser.add_argument("--sigmas", nargs="+", type=float, default=[0, 0.25, 0.5, 1.0])
parser.add_argument("--noise_seeds", type=int, default=3)
parser.add_argument("--out", default=os.path.join(RESULTS, "shift_eval.csv"))
args = parser.parse_args()

fit = BENCHMARKS["coco_val"]()
benches = {b: BENCHMARKS[b]() for b in args.benchmarks}


def evaluate(bench, img, txt):
    if "txt2img" in bench:
        return eval_retrieval(bench, img, txt)
    return eval_cases(bench, img, txt)


rows = []
for m in args.models:
    fit_img, fit_txt, _ = get_embeddings(m, fit)
    embs = {b: get_embeddings(m, bench)[:2] for b, bench in benches.items()}
    for stat in args.stats:
        delta = gap_vector(fit_img, fit_txt, stat)
        for mode in args.modes:
            settings = [("alpha", a, a) for a in args.alphas]
            settings += [("target", t, alpha_for_target_gap(fit_img, fit_txt, delta, t, mode)) for t in args.targets]
            for kind, value, alpha in settings:
                if stat == "median" and kind == "alpha" and alpha == 0:
                    continue  # identical to the mean/alpha=0 baseline
                gap_after = remaining_gap(*apply_shift(fit_img, fit_txt, delta, alpha, mode))
                for b, bench in benches.items():
                    img, txt = apply_shift(*embs[b], delta, alpha, mode)
                    base = dict(model=m, stat=stat, mode=mode, setting=kind, value=value, alpha=alpha,
                                fit_gap_after=gap_after, bench_gap_after=remaining_gap(img, txt), benchmark=b)
                    for sigma in args.sigmas:
                        seeds = range(args.noise_seeds) if sigma > 0 else [0]
                        res = []
                        for s in seeds:
                            rng = np.random.default_rng(s)
                            res.append(evaluate(bench, add_noise(img, sigma, rng), add_noise(txt, sigma, rng)))
                        rows.append(dict(base, sigma=sigma, **pd.DataFrame(res).mean().to_dict()))
                print(f"{m} {stat} {mode} {kind}={value:+.2f} (alpha={alpha:.3f}, gap on val -> {gap_after:.3f})", flush=True)

pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"wrote {args.out}")
