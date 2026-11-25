"""Tables, figures and a written summary from the final GPU run.

Usage: .venv/bin/python scripts/final_analysis.py --root final_results
Reads <root>/results/{finetune_runs,seed_variance,shift_eval,gap_metrics,naturalness,
sugarcrepe_subsets}.csv and writes <root>/analysis/: *.csv tables, fig_*.png,
summary.md. Each section is independent: a missing input skips that section only.
"""
import argparse
import os
import re
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="final_results")
a = p.parse_args()
R = os.path.join(a.root, "results")
OUT = os.path.join(a.root, "analysis")
os.makedirs(OUT, exist_ok=True)

# fixed categorical order (reference palette, light mode); colour follows the entity
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["clip", "negclip", "negclip_gelu", "ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]
COLOR = dict(zip(MODELS, PALETTE))
MARK = dict(zip(MODELS, ["o", "s", "D", "^", "v", "<", ">", "P"]))
CONFIGS = ["ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]
LABEL = {"clip": "CLIP", "negclip": "NegCLIP (released)", "negclip_gelu": "NegCLIP (GELU)",
         "ft_plain": "FT plain", "ft_plain512": "FT plain, batch 512", "ft_hardcap": "FT + hard captions",
         "ft_hardimg": "FT + hard images", "ft_negclip": "FT NegCLIP recipe"}
CLEAN = [("coco_test", "i2t_R@1", "COCO i2t R@1"), ("coco_test", "t2i_R@1", "COCO t2i R@1"),
         ("coco_order", "acc", "COCO-Order"), ("vg_relation", "macro_acc", "VG-Relation"),
         ("vg_attribution", "macro_acc", "VG-Attribution")]
SHIFT_PANELS = CLEAN[2:] + [("sugarcrepe", "acc", "SugarCrepe"), ("coco_test", "i2t_R@1", "COCO i2t R@1")]
RUN_RE = re.compile(r"^(ft_\w+?)_lr([0-9.e+-]+)_s(\d+)$")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False, "figure.dpi": 140})
md = ["# Final GPU run: results summary\n"]


def section(title):
    def wrap(fn):
        md.append(f"\n## {title}\n")
        try:
            fn()
        except Exception as e:
            md.append(f"_(section skipped: {e!r})_\n")
            traceback.print_exc()
        return fn
    return wrap


def table(df, name, floatfmt=".3f"):
    df.to_csv(os.path.join(OUT, f"{name}.csv"))
    md.append(df.to_markdown(floatfmt=floatfmt) + "\n")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    md.append(f"![{name}]({name}.png)\n")


def read(name):
    return pd.read_csv(os.path.join(R, name))


def clean_wide(df):
    """seed_variance rows -> one row per model with the clean headline metrics."""
    rows = {}
    for (bench, metric, label) in CLEAN:
        d = df[(df.benchmark == bench)].set_index("model")[metric]
        for m, v in d.items():
            rows.setdefault(m, {})[label] = v
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- training ----
@section("Fine-tuning runs: validation gap and R@1 over training (seed 0)")
def _():
    df = read("finetune_runs.csv")
    df = df[df.seed == 0]
    lrs = sorted(df.lr.unique(), reverse=True)
    fig, axes = plt.subplots(2, len(lrs), figsize=(3.4 * len(lrs), 5.2), sharey="row", squeeze=False)
    for j, lr in enumerate(lrs):
        for cfg in CONFIGS:
            d = df[(df.config == cfg) & (df.lr == lr)].sort_values("epoch")
            if d.empty:
                continue
            for i, col in enumerate(["val_gap_norm", "val_mean_R@1"]):
                axes[i, j].plot(d.epoch, d[col], marker=MARK[cfg], ms=4, color=COLOR[cfg], label=LABEL[cfg])
        axes[0, j].set_title(f"lr = {lr:g}")
        axes[1, j].set_xlabel("epoch")
    axes[0, 0].set_ylabel("COCO-val gap ||Δ||")
    axes[1, 0].set_ylabel("COCO-val mean R@1")
    axes[0, -1].legend(fontsize=7, loc="best")
    save(fig, "fig_training_curves")
    last = df.sort_values("epoch").groupby("run").tail(1)
    t = last.pivot_table(index="config", columns="lr", values=["val_mean_R@1", "val_gap_norm"])
    table(t.reindex([c for c in CONFIGS if c in t.index]), "table_val_final_epoch")


# ---------------------------------------------------------------- matched LR ----
@section("Matched-LR comparison (seed 0, clean accuracy, alpha = 0)")
def _():
    sv = read("seed_variance.csv")
    w = clean_wide(sv)
    rows = []
    for m, r in w.iterrows():
        mm = RUN_RE.match(m)
        if mm and mm.group(3) == "0":
            rows.append(dict(config=mm.group(1), lr=float(mm.group(2)), **r.to_dict()))
    t = pd.DataFrame(rows).set_index(["config", "lr"]).sort_index()
    base = w.loc[[m for m in ["clip", "negclip"] if m in w.index]]
    table(t, "table_matched_lr")
    md.append("Reference (no fine-tuning / released NegCLIP):\n\n")
    table(base, "table_reference_models")
    labels = [c[2] for c in CLEAN]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 3.0), squeeze=False)
    for k, lab in enumerate(labels):
        ax = axes[0, k]
        for cfg in CONFIGS:
            if cfg not in t.index.get_level_values(0):
                continue
            d = t.loc[cfg][lab].sort_index()
            ax.plot(d.index, d.values, marker=MARK[cfg], ms=5, color=COLOR[cfg], label=LABEL[cfg])
        for ref in base.index:
            ax.axhline(base.loc[ref, lab], color=COLOR[ref], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_title(lab)
        ax.set_xlabel("learning rate")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Clean accuracy at matched learning rates (dashed: CLIP / released NegCLIP)", y=1.02)
    save(fig, "fig_matched_lr")


# ---------------------------------------------------------------- seeds ----
@section("Seed variance (seeds 0, 1, 2; mean ± std)")
def _():
    sv = read("seed_variance.csv")
    w = clean_wide(sv)
    rows = []
    for m, r in w.iterrows():
        mm = RUN_RE.match(m)
        if mm:
            rows.append(dict(config=mm.group(1), lr=float(mm.group(2)), seed=int(mm.group(3)), **r.to_dict()))
    d = pd.DataFrame(rows)
    d = d[d.lr.isin([5e-6, 1e-6])]
    labels = [c[2] for c in CLEAN]
    g = d.groupby(["config", "lr"])[labels]
    mean, std, n = g.mean(), g.std(), g.size()
    fmt = mean.copy().astype(object)
    for c in labels:
        fmt[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c].fillna(0))]
    fmt["n seeds"] = n
    table(fmt, "table_seed_variance")
    fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 3.2), squeeze=False)
    for k, lab in enumerate(labels):
        ax = axes[0, k]
        for j, lr in enumerate([5e-6, 1e-6]):
            for i, cfg in enumerate([c for c in CONFIGS if (c, lr) in mean.index]):
                x = j * (len(CONFIGS) + 1) + i
                # points, not bars: the axis is zoomed, and a bar would imply a zero baseline
                ax.errorbar(x, mean.loc[(cfg, lr), lab], yerr=std.loc[(cfg, lr), lab], color=COLOR[cfg],
                            marker=MARK[cfg], ms=7, capsize=3, lw=1.5, ls="none",
                            label=LABEL[cfg] if (k == 0 and j == 0) else None)
        ax.set_xticks([2, len(CONFIGS) + 3])
        ax.set_xticklabels(["lr 5e-6", "lr 1e-6"])
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
    fig.legend(loc="lower center", ncol=len(CONFIGS), fontsize=7, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_seed_variance")


# ---------------------------------------------------------------- batch control ----
@section("Batch-size control: is the hard-image effect just a bigger batch?")
def _():
    sv = read("seed_variance.csv")
    w = clean_wide(sv)
    ft = read("finetune_runs.csv")
    last = ft.sort_values("epoch").groupby("run").tail(1).set_index("run")
    rows = []
    for cfg in ["ft_plain", "ft_plain512", "ft_hardimg"]:
        for lr in [5e-6, 1e-6]:
            m = f"{cfg}_lr{lr:g}_s0"
            if m in w.index:
                rows.append(dict(config=cfg, lr=lr, val_gap=last.loc[m, "val_gap_norm"] if m in last.index else np.nan,
                                 **w.loc[m].to_dict()))
    table(pd.DataFrame(rows).set_index(["config", "lr"]), "table_batch_control")


# ---------------------------------------------------------------- gap metrics ----
@section("Gap geometry of the core models (Approach 01)")
def _():
    g = read("gap_metrics.csv")
    cols = [c for c in ["model", "split", "gap_norm", "centroid_cos", "matched_cos", "var_frac_img", "var_frac_txt",
                        "linear_sep_acc"] if c in g.columns]
    t = g[cols]
    if "split" in t.columns:
        t = t[t.split == "coco_test"]
    table(t.set_index("model").reindex([m for m in MODELS if m in set(t.model)]), "table_gap_metrics")


# ---------------------------------------------------------------- shift ----
def shift_df():
    df = read("shift_eval.csv")
    return df[df.sigma == 0]


@section("Closing the gap post hoc (mode text, mean Δ, no noise): accuracy vs remaining gap")
def _():
    df = shift_df()
    d = df[(df["mode"] == "text") & (df.stat == "mean")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), squeeze=False)
    for ax, (bench, metric, lab) in zip(axes.flat, SHIFT_PANELS):
        for m in MODELS:
            dm = d[(d.model == m) & (d.benchmark == bench)].sort_values("fit_gap_after")
            if dm.empty:
                continue
            ax.plot(dm.fit_gap_after, dm[metric], marker=MARK[m], ms=3.5, lw=1.6, color=COLOR[m], label=LABEL[m])
        ax.set_title(lab)
        ax.set_xlabel("remaining gap on COCO val")
    spare = axes.flat[-1]  # 5 panels on a 2x3 grid: the last slot holds the legend
    spare.axis("off")
    spare.legend(*axes[0, 0].get_legend_handles_labels(), loc="center", fontsize=8)
    save(fig, "fig_shift_text_mean")

    # delta at the closest-to-closed setting vs alpha = 0, per model x benchmark
    rows = []
    for m in MODELS:
        dm = d[d.model == m]
        if dm.empty:
            continue
        base = dm[(dm.setting == "alpha") & (dm.alpha == 0)].set_index("benchmark")
        closed = dm[dm.setting == "target"]
        closed = closed[closed.value == closed.value.min()].set_index("benchmark")
        row = dict(model=m, gap_before=base.fit_gap_after.iloc[0], gap_after=closed.fit_gap_after.iloc[0])
        for bench, metric, lab in SHIFT_PANELS:
            if bench in base.index and bench in closed.index:
                row[f"Δ {lab}"] = closed.loc[bench, metric] - base.loc[bench, metric]
        rows.append(row)
    table(pd.DataFrame(rows).set_index("model"), "table_gap_closing_delta", floatfmt="+.3f")


@section("H3 controls: shift without re-normalisation, and orthogonalised shift")
def _():
    df = shift_df()
    d = df[(df.stat == "mean") & (df.setting == "alpha")]
    show = [m for m in ["clip", "negclip", "ft_plain", "ft_negclip"] if m in set(d.model)]
    modes = [m for m in ["text", "both", "text_norenorm", "orth"] if m in set(d["mode"])]
    mode_color = dict(zip(["text", "both", "text_norenorm", "orth"], PALETTE[:4]))
    fig, axes = plt.subplots(len(show), 3, figsize=(10, 2.6 * len(show)), squeeze=False, sharex=True)
    for r, m in enumerate(show):
        for c, (bench, metric, lab) in enumerate(CLEAN[2:]):
            ax = axes[r, c]
            for mode in modes:
                dm = d[(d.model == m) & (d.benchmark == bench) & (d["mode"] == mode)].sort_values("alpha")
                ax.plot(dm.alpha, dm[metric], marker="o", ms=3, lw=1.6, color=mode_color[mode], label=mode)
            ax.set_title(f"{LABEL[m]}: {lab}", fontsize=8)
            if r == len(show) - 1:
                ax.set_xlabel("alpha")
    axes[0, 0].legend(fontsize=7)
    save(fig, "fig_h3_controls")
    rows = []
    for m in sorted(set(d.model), key=MODELS.index):
        for mode in modes:
            dm = d[(d.model == m) & (d["mode"] == mode)]
            for bench, metric, lab in CLEAN[2:] + [("sugarcrepe", "acc", "SugarCrepe")]:
                x = dm[dm.benchmark == bench].set_index("alpha")[metric]
                if 0 in x.index and len(x) > 1:
                    rows.append(dict(model=m, mode=mode, benchmark=lab, max_abs_change=float((x - x.loc[0]).abs().max()),
                                     change_at_alpha1=float(x.get(1.0, np.nan) - x.loc[0])))
    t = pd.DataFrame(rows).pivot_table(index=["model", "mode"], columns="benchmark", values="change_at_alpha1")
    md.append("Accuracy change at alpha = 1 vs alpha = 0 (text_norenorm must be ~0 by construction):\n\n")
    table(t, "table_h3_controls", floatfmt="+.4f")
    # orth already intervenes at alpha = 0: it removes every caption's own deviation along the
    # gap direction. Its alpha = 0 cost isolates how much accuracy that 1-D component carries.
    rows = []
    for m in sorted(set(d.model), key=MODELS.index):
        for bench, metric, lab in CLEAN[2:] + [("sugarcrepe", "acc", "SugarCrepe")]:
            x = d[(d.model == m) & (d.benchmark == bench) & (d.alpha == 0)].set_index("mode")[metric]
            if {"text", "orth"} <= set(x.index):
                rows.append(dict(model=m, benchmark=lab, baseline=x["text"], without_gap_component=x["orth"],
                                 change=x["orth"] - x["text"]))
    t = pd.DataFrame(rows).pivot_table(index="model", columns="benchmark", values="change")
    md.append("Accuracy change from only removing each caption's deviation along the gap direction "
              "(orth at alpha = 0 vs the unshifted model):\n\n")
    table(t.reindex([m for m in MODELS if m in t.index]), "table_gap_component_removed", floatfmt="+.3f")


@section("Robustness to embedding noise: does a closed gap help under noise?")
def _():
    df = read("shift_eval.csv")
    d = df[(df["mode"] == "text") & (df.stat == "mean")]
    rows = []
    for m in MODELS:
        for sigma in sorted(d.sigma.unique()):
            dm = d[(d.model == m) & (d.sigma == sigma)]
            if dm.empty:
                continue
            base = dm[(dm.setting == "alpha") & (dm.alpha == 0)].set_index("benchmark")
            closed = dm[dm.setting == "target"]
            closed = closed[closed.value == closed.value.min()].set_index("benchmark")
            for bench, metric, lab in [("coco_order", "acc", "COCO-Order"), ("vg_relation", "macro_acc", "VG-Relation"),
                                       ("sugarcrepe", "acc", "SugarCrepe")]:
                if bench in base.index and bench in closed.index:
                    rows.append(dict(model=m, sigma=sigma, benchmark=lab,
                                     delta=closed.loc[bench, metric] - base.loc[bench, metric]))
    t = pd.DataFrame(rows).pivot_table(index=["benchmark", "model"], columns="sigma", values="delta")
    md.append("Accuracy change from closing the gap (smallest target) at each noise level σ:\n\n")
    table(t, "table_noise_delta", floatfmt="+.3f")


# ---------------------------------------------------------------- sugarcrepe ----
@section("SugarCrepe by subset (clean, and after closing the gap with alpha = 1, mode text)")
def _():
    s = read("sugarcrepe_subsets.csv")
    s = s[s["mode"] == "text"]
    clean = s[s.alpha == 0].pivot_table(index="model", columns="subset", values="acc")
    clean = clean.reindex([m for m in MODELS if m in clean.index])
    clean["mean"] = clean.mean(axis=1)
    table(clean, "table_sugarcrepe_clean")
    closed = s[s.alpha == 1.0].pivot_table(index="model", columns="subset", values="acc")
    delta = (closed - s[s.alpha == 0].pivot_table(index="model", columns="subset", values="acc"))
    delta = delta.reindex([m for m in MODELS if m in delta.index])
    md.append("Change after closing the gap (alpha = 1):\n\n")
    table(delta, "table_sugarcrepe_delta", floatfmt="+.3f")
    subsets = list(clean.columns.drop("mean"))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ms = list(clean.index)
    wbar = 0.8 / len(ms)
    for i, m in enumerate(ms):
        ax.bar(np.arange(len(subsets)) + (i - len(ms) / 2 + 0.5) * wbar, clean.loc[m, subsets], width=wbar * 0.92,
               color=COLOR[m], label=LABEL[m])
    ax.set_xticks(np.arange(len(subsets)))
    ax.set_xticklabels(subsets)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("accuracy (chance 0.5)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_sugarcrepe")


# ---------------------------------------------------------------- naturalness ----
@section("Caption naturalness: projection on the gap direction vs GPT-2 perplexity (COCO-Order)")
def _():
    n = read("naturalness.csv").set_index("model")
    n = n.reindex([m for m in MODELS if m in n.index])
    table(n, "table_naturalness")
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh([LABEL[m] for m in n.index], n.spearman_rho, color=[COLOR[m] for m in n.index], height=0.6)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("Spearman ρ (projection on Δ̂, GPT-2 perplexity)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    save(fig, "fig_naturalness")


with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("\n".join(md))
print("wrote", OUT)
