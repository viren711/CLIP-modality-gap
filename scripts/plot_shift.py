"""Plot metric vs remaining gap from results/shift_eval.csv.

One figure per (mode, stat): rows = benchmarks, columns = noise level sigma,
one line per model. x-axis = remaining gap on COCO val after the shift
(so models are compared at equal gap, not equal alpha).

Usage: .venv/bin/python scripts/plot_shift.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import FIG_DIR, RESULTS  # noqa: E402

# headline metric per benchmark
METRIC = {"coco_test": ["i2t_R@1", "t2i_R@1"], "coco_order": ["acc"],
          "vg_relation": ["macro_acc"], "vg_attribution": ["macro_acc"]}

df = pd.read_csv(os.path.join(RESULTS, "shift_eval.csv"))
os.makedirs(FIG_DIR, exist_ok=True)
panels = [(b, m) for b in METRIC if b in set(df.benchmark) for m in METRIC[b]]
sigmas = sorted(df.sigma.unique())

for (mode, stat), d in df.groupby(["mode", "stat"]):
    fig, axes = plt.subplots(len(panels), len(sigmas), figsize=(3.6 * len(sigmas), 2.8 * len(panels)),
                             squeeze=False, sharex=True)
    for r, (bench, metric) in enumerate(panels):
        for c, sigma in enumerate(sigmas):
            ax = axes[r, c]
            for model, dm in d[(d.benchmark == bench) & (d.sigma == sigma)].groupby("model"):
                dm = dm.sort_values("fit_gap_after")
                ax.plot(dm.fit_gap_after, dm[metric], marker="o", ms=3, label=model)
            if r == 0:
                ax.set_title(f"sigma = {sigma}")
            if c == 0:
                ax.set_ylabel(f"{bench}\n{metric}")
            if r == len(panels) - 1:
                ax.set_xlabel("remaining gap (COCO val)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"shift mode = {mode}, gap vector = {stat}")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"shift_{mode}_{stat}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
