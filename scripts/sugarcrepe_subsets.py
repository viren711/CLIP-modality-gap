"""Per-subset SugarCrepe accuracy (replace/swap/add x obj/att/rel), clean and after gap shifts.

The pooled SugarCrepe benchmark in shift_eval.py reports only acc / macro_acc;
this splits it by subset, for alpha in --alphas, shift modes text and both,
gap vector fitted on COCO val (mean), no noise.

Usage: .venv/bin/python scripts/sugarcrepe_subsets.py --models clip negclip ft_negclip
Writes results/sugarcrepe_subsets.csv.
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
from src.evaluate import case_scores  # noqa: E402
from src.gap import apply_shift, gap_vector, remaining_gap  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["clip", "negclip"])
parser.add_argument("--alphas", nargs="+", type=float, default=[-0.5, 0, 0.5, 1.0])
parser.add_argument("--out", default=os.path.join(RESULTS, "sugarcrepe_subsets.csv"))
args = parser.parse_args()

fit, bench = BENCHMARKS["coco_val"](), BENCHMARKS["sugarcrepe"]()
groups = np.array(bench["groups"])
rows = []
for m in args.models:
    fit_img, fit_txt, _ = get_embeddings(m, fit)
    img0, txt0, _ = get_embeddings(m, bench)
    delta = gap_vector(fit_img, fit_txt)
    for mode in ["text", "both"]:
        for alpha in args.alphas:
            img, txt = apply_shift(img0, txt0, delta, alpha, mode)
            correct = case_scores(bench, img, txt).argmax(1) == bench["correct"]
            gap = remaining_gap(*apply_shift(fit_img, fit_txt, delta, alpha, mode))
            for g in np.unique(groups):
                rows.append(dict(model=m, mode=mode, alpha=alpha, fit_gap_after=gap, subset=g,
                                 acc=float(correct[groups == g].mean()), n=int((groups == g).sum())))
    print(m, "done", flush=True)

pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"wrote {args.out}")
