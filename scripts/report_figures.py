"""Extra figures for the final report, built from the tables in final_results/analysis.

Usage: .venv/bin/python scripts/report_figures.py --root final_results
Writes final_results/report/figs/*.pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="final_results")
a = p.parse_args()
A = os.path.join(a.root, "analysis")
OUT = os.path.join(a.root, "report", "figs")
os.makedirs(OUT, exist_ok=True)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["clip", "negclip", "negclip_gelu", "ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]
COLOR = dict(zip(MODELS, PALETTE))
MARK = dict(zip(MODELS, ["o", "s", "D", "^", "v", "<", ">", "P"]))
LABEL = {"clip": "CLIP", "negclip": "NegCLIP (released)", "negclip_gelu": "NegCLIP (GELU)",
         "ft_plain": "FT plain", "ft_plain512": "FT plain, batch 512", "ft_hardcap": "FT + hard captions",
         "ft_hardimg": "FT + hard images", "ft_negclip": "FT NegCLIP recipe"}
SHORT = {"clip": "CLIP", "negclip": "NegCLIP", "negclip_gelu": "NegCLIP-GELU", "ft_plain": "FT plain",
         "ft_plain512": "FT 512", "ft_hardimg": "FT +img", "ft_hardcap": "FT +cap", "ft_negclip": "FT NegCLIP"}
SHOW = ["clip", "negclip", "ft_plain", "ft_plain512", "ft_hardimg", "ft_hardcap", "ft_negclip"]
INK, MUTED = "#222222", "#666666"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False, "font.family": "serif", "axes.edgecolor": "#888888",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def hbars(ax, models, vals, title):
    y = np.arange(len(models))[::-1]
    ax.barh(y, vals, color=[COLOR[m] for m in models], height=0.62)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in models])
    ax.set_title(title, fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    span = max(abs(np.min(vals)), abs(np.max(vals)))
    for yi, v in zip(y, vals):
        ax.text(v + (0.012 * span * 4 if v >= 0 else -0.012 * span * 4), yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    ax.set_xlim(-span * 1.7 if np.min(vals) < 0 else -span * 0.1, span * 1.7 if np.max(vals) > 0 else span * 0.1)


# 1. accuracy change from closing the gap
d = pd.read_csv(os.path.join(A, "table_gap_closing_delta.csv")).set_index("model")
cols = [("Δ COCO-Order", "COCO-Order"), ("Δ VG-Relation", "VG-Relation"),
        ("Δ VG-Attribution", "VG-Attribution"), ("Δ COCO i2t R@1", "COCO image→text R@1")]
fig, axs = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
for ax, (c, t) in zip(axs, cols):
    hbars(ax, SHOW, d.loc[SHOW, c].values, t)
    ax.set_xlabel("change in accuracy")
fig.tight_layout()
save(fig, "gap_closing_bars.pdf")

# 2. mechanism: removing the deviation along u vs closing the gap
orth0 = pd.read_csv(os.path.join(A, "table_gap_component_removed.csv")).set_index("model")
h3 = pd.read_csv(os.path.join(A, "table_h3_controls.csv"))
text1 = h3[h3["mode"] == "text"].set_index("model")
norenorm = h3[h3["mode"] == "text_norenorm"].set_index("model")
fig, axs = plt.subplots(1, 2, figsize=(10, 2.9))
for ax, b in zip(axs, ["COCO-Order", "VG-Relation"]):
    x = np.arange(len(SHOW))
    w = 0.27
    ax.bar(x - w, norenorm.loc[SHOW, b], w, color="#bbbbbb", label="shift without re-normalisation (α = 1)")
    ax.bar(x, orth0.loc[SHOW, b], w, color="#6250d6", label="remove deviation along û only (no shift)")
    ax.bar(x + w, text1.loc[SHOW, b], w, color="#eb6834", label="close the gap (text shift, α = 1)")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in SHOW], fontsize=7.5, rotation=25, ha="right")
    ax.set_title(b, fontsize=9, color=INK)
    ax.set_ylabel("change in accuracy")
    ax.grid(axis="x", visible=False)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
save(fig, "mechanism_bars.pdf")

# 3. noise: change from closing the gap vs noise level
nz = pd.read_csv(os.path.join(A, "table_noise_delta.csv"))
sig = [0.0, 0.25, 0.5, 1.0]
fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True)
for ax, b in zip(axs, ["COCO-Order", "VG-Relation", "SugarCrepe"]):
    sub = nz[nz.benchmark == b].set_index("model")
    for m in SHOW:
        ax.plot(sig, sub.loc[m, [str(s) for s in sig]].values, marker=MARK[m], ms=4.5, color=COLOR[m],
                label=LABEL[m], lw=1.6)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_title(b, fontsize=9, color=INK)
    ax.set_xlabel("embedding noise σ")
    ax.set_xticks(sig)
axs[0].set_ylabel("change from closing the gap")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
save(fig, "noise_lines.pdf")

# 4. SugarCrepe per-subset change: diverging heatmap
sc = pd.read_csv(os.path.join(A, "table_sugarcrepe_delta.csv")).set_index("model").loc[SHOW]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
lim = np.abs(sc.values).max()
im = ax.imshow(sc.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(sc.shape[1]))
ax.set_xticklabels([c.replace("_", "\n") for c in sc.columns])
ax.set_yticks(range(len(SHOW)))
ax.set_yticklabels([LABEL[m] for m in SHOW])
ax.grid(False)
for i in range(sc.shape[0]):
    for j in range(sc.shape[1]):
        v = sc.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 * lim else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("change after closing the gap")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
save(fig, "sugarcrepe_heatmap.pdf")

# 5. gap geometry of the core models
g = pd.read_csv(os.path.join(A, "table_gap_metrics.csv"))
g = g[g.split == "coco_test"].set_index("model").loc[SHOW]
fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
for ax, (c, t, ref) in zip(axs, [("gap_norm", "gap size ‖Δ‖ (COCO test)", None),
                                  ("centroid_cos", "cosine between modality centroids", None),
                                  ("var_frac_txt", "text variance along û", 1 / 512)]):
    y = np.arange(len(SHOW))[::-1]
    ax.barh(y, g[c], color=[COLOR[m] for m in SHOW], height=0.62)
    for yi, v in zip(y, g[c]):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7, color=INK)
    if ref:
        ax.axvline(ref, color=MUTED, ls="--", lw=0.9)
        pass
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in SHOW])
    ax.set_title(t, fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, g[c].max() * 1.3)
fig.tight_layout()
save(fig, "gap_geometry.pdf")

# 6. compositional accuracy vs final gap (all seed-0 runs): gap size does not predict skill
mlr = pd.read_csv(os.path.join(A, "table_matched_lr.csv"))
vf = pd.read_csv(os.path.join(A, "table_val_final_epoch.csv"), header=[0, 1], index_col=0).dropna(how="all")
gap = {(cfg, float(lr)): vf.loc[cfg, ("val_gap_norm", lr)] for cfg in vf.index
       for lr in vf["val_gap_norm"].columns if pd.notna(vf.loc[cfg, ("val_gap_norm", lr)])}
LRM = {1e-6: "o", 5e-6: "s", 1e-5: "^"}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), sharex=True)
for ax, b in zip(axs, ["COCO-Order", "VG-Relation"]):
    for _, r in mlr.iterrows():
        k = (r.config, float(r.lr))
        if k in gap:
            ax.scatter(gap[k], r[b], color=COLOR[r.config], marker=LRM[float(r.lr)], s=40, zorder=3,
                       edgecolor="white", linewidth=0.8)
    ax.set_xlabel("final validation gap ‖Δ‖")
    ax.set_title(b, fontsize=9, color=INK)
axs[0].set_ylabel("accuracy (clean)")
from matplotlib.lines import Line2D  # noqa: E402
cfgs = ["ft_plain", "ft_plain512", "ft_hardimg", "ft_hardcap", "ft_negclip"]
hs = [Line2D([], [], color=COLOR[c], marker="o", ls="", label=LABEL[c]) for c in cfgs]
hs += [Line2D([], [], color=MUTED, marker=m, ls="", label=f"lr {lr:g}") for lr, m in LRM.items()]
fig.legend(handles=hs, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
fig.tight_layout()
save(fig, "gap_vs_skill.pdf")
