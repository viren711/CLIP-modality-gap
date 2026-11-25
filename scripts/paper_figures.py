"""Figures for the ICML-format final report (vector PDF, Times-like fonts, no in-figure titles).

Usage: .venv/bin/python scripts/paper_figures.py --root final_results --out final_submission/report/figs
Text is converted to outlines with Ghostscript afterwards, so the PDFs embed no
non-Type-1 fonts (the course template allows Type-1 fonts only).
"""
import argparse
import os
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="final_results")
p.add_argument("--out", default="final_submission/report/figs")
a = p.parse_args()
R, A, OUT = os.path.join(a.root, "results"), os.path.join(a.root, "analysis"), a.out
os.makedirs(OUT, exist_ok=True)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["clip", "negclip", "negclip_gelu", "ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]
COLOR = dict(zip(MODELS, PALETTE))
MARK = dict(zip(MODELS, ["o", "s", "D", "^", "v", "<", ">", "P"]))
LABEL = {"clip": "CLIP", "negclip": "NegCLIP (released)", "negclip_gelu": "NegCLIP (GELU)",
         "ft_plain": "FT plain", "ft_plain512": "FT plain, batch 512", "ft_hardcap": "FT + hard captions",
         "ft_hardimg": "FT + hard images", "ft_negclip": "FT NegCLIP recipe"}
SHORT = {"clip": "CLIP", "negclip": "NegCLIP", "negclip_gelu": "NegCLIP-GELU", "ft_plain": "FT plain",
         "ft_plain512": "FT 512", "ft_hardimg": "FT+img", "ft_hardcap": "FT+cap", "ft_negclip": "FT NegCLIP"}
CONFIGS = ["ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]
SHOW = ["clip", "negclip", "ft_plain", "ft_plain512", "ft_hardimg", "ft_hardcap", "ft_negclip"]
CLEAN = [("coco_test", "i2t_R@1", "COCO i2t R@1"), ("coco_test", "t2i_R@1", "COCO t2i R@1"),
         ("coco_order", "acc", "COCO-Order"), ("vg_relation", "macro_acc", "VG-Relation"),
         ("vg_attribution", "macro_acc", "VG-Attribution")]
RUN_RE = re.compile(r"^(ft_\w+?)_lr([0-9.e+-]+)_s(\d+)$")
INK, MUTED, ZERO = "#222222", "#555555", "#888888"
COL, FULL = 3.25, 6.75
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.edgecolor": ZERO, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "pdf.fonttype": 42, "axes.titlepad": 3})
GS = shutil.which("gs")


def save(fig, name):
    raw = os.path.join(OUT, f"_{name}.pdf")
    fig.savefig(raw, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    dst = os.path.join(OUT, f"{name}.pdf")
    if GS:  # text -> outlines: no TrueType/Type-3 fonts left in the figure
        subprocess.run([GS, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                        f"-sOutputFile={dst}", raw], check=True)
        os.remove(raw)
    else:
        os.replace(raw, dst)
    print("wrote", dst)


def rd(d, name):
    return pd.read_csv(os.path.join(d, name))


def clean_wide(df):
    rows = {}
    for bench, metric, label in CLEAN:
        for m, v in df[df.benchmark == bench].set_index("model")[metric].items():
            rows.setdefault(m, {})[label] = v
    return pd.DataFrame(rows).T


def runs_table():
    w = clean_wide(rd(R, "seed_variance.csv"))
    rows = []
    for m, r in w.iterrows():
        mm = RUN_RE.match(m)
        if mm:
            rows.append(dict(run=m, config=mm.group(1), lr=float(mm.group(2)), seed=int(mm.group(3)), **r.to_dict()))
    return pd.DataFrame(rows), w


ft = rd(R, "finetune_runs.csv")
last = ft.sort_values("epoch").groupby("run").tail(1).set_index("run")
runs, wide = runs_table()
LRM = {1e-6: "o", 5e-6: "s", 1e-5: "^"}


def lr_tex(lr):
    return {1e-6: r"$10^{-6}$", 5e-6: r"$5\times10^{-6}$", 1e-5: r"$10^{-5}$"}[lr]


# ------------------------------------------------------------------ main body ----
# M1: change from closing the gap (4 panels, full width)
d = rd(A, "table_gap_closing_delta.csv").set_index("model")
cols = [("Δ COCO-Order", "COCO-Order"), ("Δ VG-Relation", "VG-Relation"),
        ("Δ VG-Attribution", "VG-Attribution"), ("Δ COCO i2t R@1", r"COCO image$\to$text R@1")]
fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.75), sharey=True)
for ax, (c, t) in zip(axs, cols):
    v = d.loc[SHOW, c].values
    y = np.arange(len(SHOW))[::-1]
    ax.barh(y, v, color=[COLOR[m] for m in SHOW], height=0.66)
    ax.axvline(0, color=ZERO, lw=0.7)
    span = np.abs(v).max()
    for yi, vi in zip(y, v):
        ax.text(vi + (0.04 if vi >= 0 else -0.04) * span, yi, f"{vi:+.3f}", va="center",
                ha="left" if vi >= 0 else "right", fontsize=6, color=INK)
    lo = -span * 1.75 if v.min() < 0 else -span * 0.15
    hi = span * 1.75 if v.max() > 0 else span * 0.15
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"change in {t}")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in SHOW])
    ax.grid(axis="y", visible=False)
fig.tight_layout(w_pad=0.6)
save(fig, "closing_bars")

# M2: (a) validation gap over training at lr 5e-6, (b) COCO-Order vs final gap for all seed-0 runs
fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.75))
for cfg in CONFIGS:
    dd = ft[(ft.config == cfg) & (ft.lr == 5e-6) & (ft.seed == 0)].sort_values("epoch")
    a1.plot(dd.epoch, dd.val_gap_norm, marker=MARK[cfg], ms=2.8, color=COLOR[cfg], lw=1.1)
a1.set_xlabel("epoch")
a1.set_ylabel(r"validation gap $\|\Delta\|$")
a1.set_xticks(range(6))
s0 = runs[runs.seed == 0]
for _, r in s0.iterrows():
    if r.run in last.index:
        a2.scatter(last.loc[r.run, "val_gap_norm"], r["COCO-Order"], color=COLOR[r.config], marker=LRM[r.lr],
                   s=14, edgecolor="white", linewidth=0.4, zorder=3)
a2.set_xlabel(r"final validation gap $\|\Delta\|$")
a2.set_ylabel("COCO-Order accuracy")
hs = [Line2D([], [], color=COLOR[c], marker=MARK[c], ms=3.5, lw=1.1, label=LABEL[c]) for c in CONFIGS]
fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), columnspacing=0.8,
           handlelength=1.6)
fig.tight_layout(w_pad=0.8)
save(fig, "training_gap")

# M3: mechanism, COCO-Order (column)
orth0 = rd(A, "table_gap_component_removed.csv").set_index("model")
h3 = rd(A, "table_h3_controls.csv")
text1 = h3[h3["mode"] == "text"].set_index("model")
noren = h3[h3["mode"] == "text_norenorm"].set_index("model")


def mech_panel(ax, b, ticks=True):
    x = np.arange(len(SHOW))
    w = 0.27
    ax.bar(x - w, noren.loc[SHOW, b], w, color="#9a9a9a", label=r"shift, no re-normalisation ($\alpha=1$)")
    ax.bar(x, orth0.loc[SHOW, b], w, color="#6250d6", label=r"remove deviation along $\hat u$ only")
    ax.bar(x + w, text1.loc[SHOW, b], w, color="#eb6834", label=r"close the gap ($\alpha=1$)")
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in SHOW], rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"change in {b}")


fig, ax = plt.subplots(figsize=(COL, 2.0))
mech_panel(ax, "COCO-Order")
h, l = ax.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.8))
fig.legend(h, l, loc="upper center", ncol=1, fontsize=6.2, handlelength=1.2, bbox_to_anchor=(0.55, 1.0))
save(fig, "mechanism")

# M4: SugarCrepe per-subset change (column heatmap)
sc = rd(A, "table_sugarcrepe_delta.csv").set_index("model").loc[SHOW]
fig, ax = plt.subplots(figsize=(COL, 1.95))
lim = np.abs(sc.values).max()
im = ax.imshow(sc.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(sc.shape[1]))
ax.set_xticklabels([c.replace("replace", "repl").replace("_", "\n") for c in sc.columns], fontsize=6.3)
ax.set_yticks(range(len(SHOW)))
ax.set_yticklabels([LABEL[m] for m in SHOW], fontsize=6.3)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(sc.shape[0]):
    for j in range(sc.shape[1]):
        v = sc.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 0.6 * lim else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.ax.tick_params(labelsize=6)
cb.set_label("accuracy change", fontsize=6.5)
fig.tight_layout()
save(fig, "sugarcrepe_delta")

# ------------------------------------------------------------------ appendix ----
# A1: training curves, all learning rates
lrs = [1e-5, 5e-6, 1e-6]
fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.3), sharey="row", sharex=True)
for j, lr in enumerate(lrs):
    for cfg in CONFIGS:
        dd = ft[(ft.config == cfg) & (ft.lr == lr) & (ft.seed == 0)].sort_values("epoch")
        if dd.empty:
            continue
        for i, col in enumerate(["val_gap_norm", "val_mean_R@1"]):
            axes[i, j].plot(dd.epoch, dd[col], marker=MARK[cfg], ms=3, color=COLOR[cfg], label=LABEL[cfg])
    axes[0, j].text(0.97, 0.95, f"lr {lr_tex(lr)}", transform=axes[0, j].transAxes, ha="right", va="top", color=INK)
    axes[1, j].set_xlabel("epoch")
axes[0, 0].set_ylabel(r"validation gap $\|\Delta\|$")
axes[1, 0].set_ylabel("validation mean R@1")
fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "app_training_curves")

# A2: matched learning rates
mlr = runs[runs.seed == 0].set_index(["config", "lr"]).sort_index()
fig, axes = plt.subplots(1, 5, figsize=(FULL, 1.75))
for ax, (_, _, lab) in zip(axes, CLEAN):
    for cfg in CONFIGS:
        dd = mlr.loc[cfg][lab].sort_index()
        ax.plot(dd.index, dd.values, marker=MARK[cfg], ms=3, color=COLOR[cfg], label=LABEL[cfg])
    for ref in ["clip", "negclip"]:
        ax.axhline(wide.loc[ref, lab], color=COLOR[ref], lw=0.9, ls="--", label=f"{LABEL[ref]} (reference)")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(lab, labelpad=1)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.4)
save(fig, "app_matched_lr")

# A3: seed variance
sv = runs[runs.lr.isin([5e-6, 1e-6])]
labels = [c[2] for c in CLEAN]
g = sv.groupby(["config", "lr"])[labels]
mean, std = g.mean(), g.std().fillna(0)
fig, axes = plt.subplots(1, 5, figsize=(FULL, 1.75))
for k, (ax, lab) in enumerate(zip(axes, labels)):
    for j, lr in enumerate([5e-6, 1e-6]):
        for i, cfg in enumerate(CONFIGS):
            if (cfg, lr) not in mean.index:
                continue
            ax.errorbar(j * (len(CONFIGS) + 1) + i, mean.loc[(cfg, lr), lab], yerr=std.loc[(cfg, lr), lab],
                        color=COLOR[cfg], marker=MARK[cfg], ms=3.5, capsize=2, lw=1, ls="none",
                        label=LABEL[cfg] if (k == 0 and j == 0) else None)
    ax.set_xticks([2, len(CONFIGS) + 3])
    ax.set_xticklabels([f"lr {lr_tex(5e-6)}", f"lr {lr_tex(1e-6)}"])
    ax.set_ylabel(lab, labelpad=1)
    ax.grid(axis="x", visible=False)
fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.4)
save(fig, "app_seed_variance")

# A4: gap geometry
gm = rd(A, "table_gap_metrics.csv")
gm = gm[gm.split == "coco_test"].set_index("model").loc[SHOW]
fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.75), sharey=True)
for ax, (c, t, ref) in zip(axs, [("gap_norm", r"gap size $\|\Delta\|$", None),
                                  ("centroid_cos", "cosine between modality centroids", None),
                                  ("var_frac_txt", r"fraction of text variance along $\hat u$", 1 / 512)]):
    y = np.arange(len(SHOW))[::-1]
    ax.barh(y, gm[c], color=[COLOR[m] for m in SHOW], height=0.66)
    for yi, v in zip(y, gm[c]):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=6, color=INK)
    if ref:
        ax.axvline(ref, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in SHOW])
    ax.set_xlabel(t)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, gm[c].max() * 1.32)
fig.tight_layout(w_pad=0.6)
save(fig, "app_gap_geometry")

# A5: compositional accuracy vs final gap, both benchmarks
fig, axs = plt.subplots(1, 2, figsize=(FULL * 0.8, 1.9))
for ax, b in zip(axs, ["COCO-Order", "VG-Relation"]):
    for _, r in s0.iterrows():
        if r.run in last.index:
            ax.scatter(last.loc[r.run, "val_gap_norm"], r[b], color=COLOR[r.config], marker=LRM[r.lr], s=20,
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel(r"final validation gap $\|\Delta\|$")
    ax.set_ylabel(f"{b} accuracy")
hs = [Line2D([], [], color=COLOR[c], marker="o", ls="", ms=4, label=LABEL[c]) for c in CONFIGS]
hs += [Line2D([], [], color=MUTED, marker=m, ls="", ms=4, label=f"lr {lr_tex(lr)}") for lr, m in LRM.items()]
fig.legend(handles=hs, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
save(fig, "app_gap_vs_skill")

# A6: accuracy vs remaining gap (text shift, mean delta, clean)
se = rd(R, "shift_eval.csv")
se0 = se[se.sigma == 0]
dt = se0[(se0["mode"] == "text") & (se0.stat == "mean")]
panels = CLEAN[2:] + [("sugarcrepe", "acc", "SugarCrepe"), ("coco_test", "i2t_R@1", r"COCO i2t R@1")]
fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.6))
for ax, (bench, metric, lab) in zip(axes.flat, panels):
    for m in MODELS:
        dm = dt[(dt.model == m) & (dt.benchmark == bench)].sort_values("fit_gap_after")
        ax.plot(dm.fit_gap_after, dm[metric], marker=MARK[m], ms=2.6, lw=1.1, color=COLOR[m], label=LABEL[m])
    ax.set_xlabel("remaining gap on COCO val")
    ax.set_ylabel(lab)
axes.flat[-1].axis("off")
axes.flat[-1].legend(*axes[0, 0].get_legend_handles_labels(), loc="center")
fig.tight_layout()
save(fig, "app_shift_curves")

# A7: four shift modes vs alpha
da = se0[(se0.stat == "mean") & (se0.setting == "alpha")]
show = ["clip", "negclip", "ft_plain", "ft_negclip"]
modes = ["text", "both", "text_norenorm", "orth"]
mcol = dict(zip(modes, PALETTE[:4]))
fig, axes = plt.subplots(4, 3, figsize=(FULL, 6.4), sharex=True)
for r, m in enumerate(show):
    for c, (bench, metric, lab) in enumerate(CLEAN[2:]):
        ax = axes[r, c]
        for mode in modes:
            dm = da[(da.model == m) & (da.benchmark == bench) & (da["mode"] == mode)].sort_values("alpha")
            ax.plot(dm.alpha, dm[metric], marker="o", ms=2.4, lw=1.1, color=mcol[mode], label=mode)
        ax.set_ylabel(f"{LABEL[m]}\n{lab}", fontsize=6.5)
        if r == 3:
            ax.set_xlabel(r"shift strength $\alpha$")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.025, 1, 1))
save(fig, "app_shift_modes")

# A8: mechanism, both benchmarks
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.0))
for ax, b in zip(axs, ["COCO-Order", "VG-Relation"]):
    mech_panel(ax, b)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
save(fig, "app_mechanism")

# A9: SugarCrepe clean accuracy per subset
scs = rd(R, "sugarcrepe_subsets.csv")
clean = scs[(scs["mode"] == "text") & (scs.alpha == 0)].pivot_table(index="model", columns="subset", values="acc")
clean = clean.reindex([m for m in MODELS if m in clean.index])
subsets = list(clean.columns)
fig, ax = plt.subplots(figsize=(FULL, 2.0))
wb = 0.8 / len(clean)
for i, m in enumerate(clean.index):
    ax.bar(np.arange(len(subsets)) + (i - len(clean) / 2 + 0.5) * wb, clean.loc[m, subsets], width=wb * 0.9,
           color=COLOR[m], label=LABEL[m])
ax.set_xticks(np.arange(len(subsets)))
ax.set_xticklabels(subsets)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("accuracy (chance 0.5)")
ax.grid(axis="x", visible=False)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "app_sugarcrepe_clean")

# A10: change from closing the gap vs noise level
nz = rd(A, "table_noise_delta.csv")
sig = [0.0, 0.25, 0.5, 1.0]
fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.9), sharex=True)
for ax, b in zip(axs, ["COCO-Order", "VG-Relation", "SugarCrepe"]):
    sub = nz[nz.benchmark == b].set_index("model")
    for m in SHOW:
        ax.plot(sig, sub.loc[m, [str(s) for s in sig]].values, marker=MARK[m], ms=3, color=COLOR[m], label=LABEL[m])
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xlabel(r"embedding noise $\sigma$")
    ax.set_ylabel(f"change in {b}")
    ax.set_xticks(sig)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "app_noise")

# A11: naturalness
nat = rd(R, "naturalness.csv").set_index("model").reindex(MODELS)
fig, ax = plt.subplots(figsize=(COL * 1.25, 1.6))
ax.barh([LABEL[m] for m in nat.index], nat.spearman_rho, color=[COLOR[m] for m in nat.index], height=0.62)
ax.axvline(0, color=ZERO, lw=0.7)
ax.set_xlabel(r"Spearman $\rho$ between $y^\top\hat u$ and GPT-2 perplexity")
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "app_naturalness")
