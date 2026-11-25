"""Tables and figures for the extra experiments (X1-X5), from final_results/extra.

Usage: .venv/bin/python scripts/extra_report.py --root final_results --figs final_submission/report/figs
Writes final_results/extra/analysis/*.csv and <figs>/x_*.pdf
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="final_results")
p.add_argument("--figs", default="final_submission/report/figs")
a = p.parse_args()
X = os.path.join(a.root, "extra")
OUT = os.path.join(X, "analysis")
os.makedirs(OUT, exist_ok=True)
os.makedirs(a.figs, exist_ok=True)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["clip", "negclip", "negclip_gelu", "ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]
COLOR = dict(zip(MODELS, PALETTE))
LABEL = {"clip": "CLIP", "negclip": "NegCLIP (released)", "negclip_gelu": "NegCLIP (GELU)",
         "ft_plain": "FT plain", "ft_plain512": "FT plain, batch 512", "ft_hardcap": "FT + hard captions",
         "ft_hardimg": "FT + hard images", "ft_negclip": "FT NegCLIP recipe"}
SHORT = {"clip": "CLIP", "negclip": "NegCLIP", "negclip_gelu": "NegCLIP-GELU", "ft_plain": "FT plain",
         "ft_plain512": "FT 512", "ft_hardimg": "FT+img", "ft_hardcap": "FT+cap", "ft_negclip": "FT NegCLIP"}
SHOW = ["clip", "negclip", "ft_plain", "ft_plain512", "ft_hardimg", "ft_hardcap", "ft_negclip"]
BN = {"coco_order": "COCO-Order", "vg_relation": "VG-Relation", "vg_attribution": "VG-Attribution",
      "sugarcrepe": "SugarCrepe"}
INK, MUTED, ZERO = "#222222", "#555555", "#888888"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.4, "legend.frameon": False, "axes.edgecolor": ZERO,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
GS = shutil.which("gs")


def save(fig, name):
    raw = os.path.join(a.figs, f"_{name}.pdf")
    fig.savefig(raw, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    dst = os.path.join(a.figs, f"{name}.pdf")
    if GS:
        subprocess.run([GS, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                        f"-sOutputFile={dst}", raw], check=True)
        os.remove(raw)
    else:
        os.replace(raw, dst)
    print("wrote", dst)


def tab(df, name):
    df.to_csv(os.path.join(OUT, f"{name}.csv"))
    print(f"\n## {name}\n" + df.to_string(float_format=lambda v: f"{v:+.3f}" if abs(v) < 1 and v != 0 else f"{v:.3f}"))


J = {}
for f in glob.glob(os.path.join(X, "results", "extra", "*.json")):
    d = json.load(open(f))
    J[d["model"]] = d
core = [m for m in MODELS if m in J]

# ---- X2 direction removal ----------------------------------------------------------------
rows = []
for m in core:
    for b, r in J[m]["x2_removal"].items():
        rows.append(dict(model=m, benchmark=BN[b], base=r["base"], u=r["u"], random_mean=r["random_mean"],
                         random_min=r["random_min"], random_max=r["random_max"], pc1=r["pc1"], pc5=r["pc5"],
                         pc20=r["pc20"]))
x2 = pd.DataFrame(rows)
if len(x2):
    tab(x2.set_index(["benchmark", "model"]).sort_index(), "x2_direction_removal")
    tab(pd.DataFrame({m: dict(var_u=J[m]["var_frac_txt_u"], var_pc1=J[m]["var_frac_txt_pc1"]) for m in core}).T,
        "x2_variance")
    fig, axs = plt.subplots(1, 2, figsize=(6.75, 1.9))
    for ax, b in zip(axs, ["COCO-Order", "VG-Attribution"]):
        d = x2[x2.benchmark == b].set_index("model").loc[[m for m in SHOW if m in core]]
        x = np.arange(len(d))
        w = 0.2
        ax.bar(x - 1.5 * w, d.u, w, color="#eb6834", label=r"gap direction $\hat u$")
        ax.bar(x - 0.5 * w, d.random_mean, w, color="#9a9a9a", label="random direction (mean of 20)")
        ax.errorbar(x - 0.5 * w, d.random_mean, yerr=[d.random_mean - d.random_min, d.random_max - d.random_mean],
                    fmt="none", ecolor=INK, lw=0.6, capsize=1.5)
        ax.bar(x + 0.5 * w, d.pc1, w, color="#2a78d6", label="top text principal component")
        ax.bar(x + 1.5 * w, d.pc5, w, color="#1baf7a", label="top 5 principal components")
        ax.axhline(0, color=ZERO, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in d.index], rotation=30, ha="right")
        ax.set_ylabel(f"change in {b}")
        ax.grid(axis="x", visible=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, "x_direction_removal")

# ---- X3 probe ------------------------------------------------------------------------------
rows = []
for m in core:
    for b, r in J[m]["x3_probe"].items():
        rows.append(dict(model=m, benchmark=BN[b], auc_true_gt_neg=r["auc_true_gt_neg"],
                         true_is_max_d=r["true_is_max_d"], true_is_min_d=r["true_is_min_d"], chance=r["chance"],
                         mean_d_true_minus_neg=r["mean_d_true_minus_neg"]))
x3 = pd.DataFrame(rows)
if len(x3):
    tab(x3.pivot_table(index="model", columns="benchmark", values="auc_true_gt_neg").reindex(core), "x3_auc")
    tab(x3.set_index(["benchmark", "model"]).sort_index(), "x3_probe")

# ---- X4 length ------------------------------------------------------------------------------
rows, rows_sc = [], []
for m in core:
    for b, r in J[m]["x4_length"].items():
        rows.append(dict(model=m, benchmark=BN[b], rho=r["rho_d_length"]))
        for g, s in r.get("by_subset", {}).items():
            rows_sc.append(dict(model=m, subset=g, **s))
if rows:
    tab(pd.DataFrame(rows).pivot_table(index="model", columns="benchmark", values="rho").reindex(core), "x4_length_rho")
    sc = pd.DataFrame(rows_sc)
    tab(sc.pivot_table(index="model", columns="subset", values="rho_dd_dl").reindex(core), "x4_sc_rho_dd_dl")
    tab(sc.pivot_table(index="model", columns="subset", values="mean_d_neg_minus_pos").reindex(core), "x4_sc_mean_dd")
    tab(sc.pivot_table(index="model", columns="subset", values="mean_len_neg_minus_pos").reindex(core), "x4_sc_mean_dlen")

# ---- X5 deviation-preserving closing ---------------------------------------------------------
rows = []
for m in core:
    for r in J[m]["x5_keep_dev"]:
        rows.append(dict(model=m, **r))
    rows[-1]["alpha"] = J[m]["x5_alpha"]
x5 = pd.DataFrame(rows)
if len(x5):
    cols = [c for c in x5.columns if ":" in c]
    tab(x5.set_index(["model", "sigma", "variant"])[cols].sort_index(), "x5_keep_dev")
    gaps = pd.DataFrame({m: J[m]["x5_gap_after"] for m in core}).T
    tab(gaps, "x5_gap_after")

# ---- X1 gap regulariser -----------------------------------------------------------------------
GR = {"ft_hardcap_gr1": ("ft_hardcap", 1), "ft_hardcap_gr10": ("ft_hardcap", 10), "ft_plain_gr10": ("ft_plain", 10),
      "ft_negclip_gr10": ("ft_negclip", 10)}
se = []
for f in glob.glob(os.path.join(X, "results", "extra", "shift_*.csv")):
    try:
        se.append(pd.read_csv(f))
    except Exception as e:
        print("skip", f, e)
logs = {}
for name in GR:
    lf = os.path.join(X, "runs", f"{name}_lr5e-06_s0", "log.jsonl")
    if os.path.exists(lf):
        logs[name] = pd.DataFrame([json.loads(line) for line in open(lf)])
if se:
    se = pd.concat(se)
    base = pd.read_csv(os.path.join(a.root, "analysis", "table_matched_lr.csv"))
    base = base[base.lr == 5e-6].set_index("config")
    ft = pd.read_csv(os.path.join(a.root, "results", "finetune_runs.csv"))
    last = ft.sort_values("epoch").groupby("run").tail(1).set_index("run")
    rows = []
    for name, (cfg, lam) in GR.items():
        d = se[(se.model == name) & (se.sigma == 0) & (se.setting == "alpha") & (se.alpha == 0)]
        if d.empty:
            continue
        g = d.set_index("benchmark")
        row = dict(model=name, config=cfg, lam=lam,
                   val_gap=logs[name].val_gap_norm.iloc[-1] if name in logs else np.nan,
                   val_gap_lam0=last.loc[f"{cfg}_lr5e-06_s0", "val_gap_norm"],
                   coco_order=g.loc["coco_order", "acc"], coco_order_lam0=base.loc[cfg, "COCO-Order"],
                   vg_rel=g.loc["vg_relation", "macro_acc"], vg_rel_lam0=base.loc[cfg, "VG-Relation"],
                   vg_attr=g.loc["vg_attribution", "macro_acc"], vg_attr_lam0=base.loc[cfg, "VG-Attribution"],
                   i2t=g.loc["coco_test", "i2t_R@1"], i2t_lam0=base.loc[cfg, "COCO i2t R@1"],
                   t2i=g.loc["coco_test", "t2i_R@1"], t2i_lam0=base.loc[cfg, "COCO t2i R@1"],
                   sugarcrepe=g.loc["sugarcrepe", "acc"] if "sugarcrepe" in g.index else np.nan)
        # post-hoc closing of what is left, for comparison
        t = se[(se.model == name) & (se.sigma == 0) & (se.setting == "target")]
        if len(t):
            t = t[t.value == t.value.min()].set_index("benchmark")
            row["posthoc_coco_order_delta"] = t.loc["coco_order", "acc"] - row["coco_order"]
            row["posthoc_i2t_delta"] = t.loc["coco_test", "i2t_R@1"] - row["i2t"]
        n = se[(se.model == name) & (se.sigma == 0.5) & (se.setting == "alpha") & (se.alpha == 0)].set_index("benchmark")
        if len(n):
            row["noisy_coco_order"] = n.loc["coco_order", "acc"]
            row["noisy_t2i"] = n.loc["coco_test", "t2i_R@1"]
        rows.append(row)
    x1 = pd.DataFrame(rows).set_index("model")
    tab(x1.T, "x1_gap_regulariser")
if logs:
    ft = pd.read_csv(os.path.join(a.root, "results", "finetune_runs.csv"))
    fig, ax = plt.subplots(figsize=(3.25, 1.7))
    for name, (cfg, lam) in GR.items():
        if name not in logs:
            continue
        b = ft[(ft.config == cfg) & (ft.lr == 5e-6) & (ft.seed == 0)].sort_values("epoch")
        ls = "-" if lam == 10 else ":"
        ax.plot(logs[name].epoch, logs[name].val_gap_norm, ls=ls, marker="o", ms=2.5, color=COLOR[cfg],
                label=fr"{LABEL[cfg]}, $\lambda$={lam}")
        if lam == 10 or cfg != "ft_hardcap":
            ax.plot(b.epoch, b.val_gap_norm, ls="--", lw=0.9, color=COLOR[cfg], alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"validation gap $\|\Delta\|$")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.2)
    fig.tight_layout()
    save(fig, "x_gapreg_curves")

# ---- X3 vs X2: the probe AUC predicts the effect of removing the coordinate ---------------------
if len(x2) and len(x3):
    m = x2.merge(x3, on=["benchmark", "model"])
    r = np.corrcoef(m.auc_true_gt_neg, m.u)[0, 1]
    tab(pd.DataFrame(dict(pearson_r=[r], n=[len(m)])), "x3_vs_x2_corr")
    BM = {"COCO-Order": "o", "VG-Relation": "s", "VG-Attribution": "^", "SugarCrepe": "D"}
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    for _, row in m.iterrows():
        ax.scatter(row.auc_true_gt_neg, row.u, color=COLOR[row.model], marker=BM[row.benchmark], s=16,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.axvline(0.5, color=ZERO, lw=0.7, ls="--")
    ax.set_xlabel(r"P($d_{\mathrm{true}} > d_{\mathrm{neg}}$) using only $d=y^\top\hat u$")
    ax.set_ylabel(r"accuracy change when $d$ is removed")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=MUTED, marker=mk, ls="", ms=3.5, label=b) for b, mk in BM.items()]
    hs += [Line2D([], [], color=COLOR[mm], marker="o", ls="", ms=3.5, label=LABEL[mm]) for mm in SHOW + ["negclip_gelu"]]
    ax.legend(handles=hs, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5.8, handletextpad=0.2)
    fig.tight_layout()
    save(fig, "x_probe_vs_removal")
