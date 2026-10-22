"""Approach 01: measure the modality gap for each model.

Usage: .venv/bin/python scripts/measure_gap.py --models clip negclip
Writes results/gap_metrics.csv and results/figures/pca_<split>.png.
Needs cached embeddings from scripts/embed_all.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.common import FIG_DIR, RESULTS  # noqa: E402
from src.data import BENCHMARKS  # noqa: E402
from src.embed import get_embeddings  # noqa: E402
from src.gap import gap_metrics  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["clip", "negclip"])
parser.add_argument("--splits", nargs="+", default=["coco_val", "coco_test"])
args = parser.parse_args()

os.makedirs(FIG_DIR, exist_ok=True)
rows = []
for split in args.splits:
    bench = BENCHMARKS[split]()
    fig, axes = plt.subplots(1, len(args.models), figsize=(5 * len(args.models), 4.5), squeeze=False)
    for ax, m in zip(axes[0], args.models):
        img, txt, scale = get_embeddings(m, bench)
        r = dict(model=m, split=split, logit_scale=scale, **gap_metrics(img, txt, bench["txt2img"]))
        rows.append(r)
        print(f"{m:8s} {split:9s} ||gap||={r['gap_norm']:.3f}  matched cos={r['matched_cos']:.3f}  "
              f"var along gap: img {r['var_frac_img']:.4f} / txt {r['var_frac_txt']:.4f} "
              f"(random {r['var_frac_random']:.4f})")

        # PCA of both modalities together, as in Liang et al. Fig. 1
        sub = np.random.default_rng(0).choice(len(txt), len(img), replace=False)
        z = PCA(2).fit_transform(np.concatenate([img, txt[sub]]))
        ax.scatter(*z[:len(img)].T, s=2, alpha=0.4, label="image")
        ax.scatter(*z[len(img):].T, s=2, alpha=0.4, label="text")
        ax.set_title(f"{m}  ||gap|| = {r['gap_norm']:.2f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"pca_{split}.png"), dpi=150)
    plt.close(fig)

out = os.path.join(RESULTS, "gap_metrics.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}")
