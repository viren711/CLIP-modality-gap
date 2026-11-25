"""Learning-rate analysis of post-hoc gap closing: reads results/lr_gap/*.csv (one per grid run) and
writes per-run and per-(config, lr) tables of the accuracy change after the text shift to gap 0.1
and after removing the deviation along u (orth, alpha=0).

Usage: python scripts/lr_gap_report.py --root final_results/lr --out final_results/lr/analysis
"""
import argparse
import glob
import os
import re

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--root", default="final_results/lr")
p.add_argument("--out", default="final_results/lr/analysis")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

METRICS = {"coco_order": "acc", "vg_relation": "macro_acc", "vg_attribution": "macro_acc", "coco_test": "i2t_R@1"}
rows = []
for f in sorted(glob.glob(os.path.join(a.root, "results/lr_gap/*.csv"))):
    d = pd.read_csv(f)
    d = d[d.sigma == 0]
    m = d.model.iloc[0]
    cfg, lr, seed = re.match(r"(.+)_lr([0-9e.-]+)_s(\d)", m).groups()
    base = d[(d["mode"] == "text") & (d.setting == "alpha") & (d.value == 0)]
    shift = d[(d["mode"] == "text") & (d.setting == "target") & (d.value == 0.1)]
    orth = d[(d["mode"] == "orth") & (d.setting == "alpha") & (d.value == 0)]
    r = dict(model=m, config=cfg, lr=float(lr), seed=int(seed), gap=float(base.fit_gap_after.iloc[0]))
    for b, col in METRICS.items():
        v0 = base[base.benchmark == b][col].iloc[0]
        r[f"{b}"] = v0
        r[f"d_shift_{b}"] = shift[shift.benchmark == b][col].iloc[0] - v0
        r[f"d_orth_{b}"] = orth[orth.benchmark == b][col].iloc[0] - v0
    rows.append(r)
df = pd.DataFrame(rows).sort_values(["config", "lr", "seed"])
df.to_csv(os.path.join(a.out, "lr_gap_runs.csv"), index=False)
num = [c for c in df.columns if c not in ("model", "config", "lr", "seed")]
g = df.groupby(["config", "lr"])[num]
summ = g.mean().join(g.std().add_suffix("_std")).join(df.groupby(["config", "lr"]).size().rename("n"))
summ.to_csv(os.path.join(a.out, "lr_gap_summary.csv"))
pd.set_option("display.width", 250)
print(summ[["n", "gap", "coco_order", "d_shift_coco_order", "d_shift_coco_order_std", "d_orth_coco_order",
            "d_shift_vg_relation", "d_shift_vg_attribution", "d_shift_coco_test"]].round(3))

# ---- figure: accuracy change from closing the gap against learning rate ----
import shutil, subprocess  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "legend.frameon": False, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555", "pdf.fonttype": 42})
CFG = [("ft_plain", "FT plain", "#008300", "^"), ("ft_hardcap", "FT + hard captions", "#2a78d6", "<"),
       ("ft_hardimg", "FT + hard images", "#eda100", ">"), ("ft_negclip", "FT NegCLIP recipe", "#e34948", "P")]
PAN = [("d_shift_coco_order", "$\\Delta$ COCO-Order"), ("d_shift_vg_attribution", "$\\Delta$ VG-Attribution"),
       ("d_shift_coco_test", "$\\Delta$ COCO i2t R@1")]
s = summ.reset_index()
fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.75))
for ax, (col, lab) in zip(axs, PAN):
    ax.axhline(0, color="#888888", lw=0.6)
    for c, name, color, mk in CFG:
        r = s[s.config == c].sort_values("lr")
        ax.errorbar(r.lr, r[col], yerr=r[col + "_std"].fillna(0), color=color, marker=mk, ms=4, capsize=2, label=name)
    ax.set_xscale("log"); ax.set_xticks([1e-6, 5e-6, 1e-5]); ax.set_xticklabels(["$10^{-6}$", "$5{\\times}10^{-6}$", "$10^{-5}$"])
    ax.minorticks_off(); ax.set_xlabel("learning rate"); ax.set_ylabel(lab)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=1.0)
outd = "final_submission/report/figs"
raw, dst = os.path.join(outd, "_lr_closing.pdf"), os.path.join(outd, "lr_closing.pdf")
fig.savefig(raw, bbox_inches="tight", pad_inches=0.02)
subprocess.run([shutil.which("gs"), "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                f"-sOutputFile={dst}", raw], check=True)
os.remove(raw)
print("wrote", dst)

# ---- LaTeX table for the appendix ----
NAMES = {"ft_plain": "FT plain", "ft_plain512": "FT plain, batch 512", "ft_hardcap": "FT + hard cap.",
         "ft_hardimg": "FT + hard img.", "ft_negclip": "FT NegCLIP"}
LRS = {1e-6: "$10^{-6}$", 5e-6: "$5{\\times}10^{-6}$", 1e-5: "$10^{-5}$"}
lines = []
for c in ["ft_plain", "ft_plain512", "ft_hardcap", "ft_hardimg", "ft_negclip"]:
    sub = s[s.config == c].sort_values("lr")
    for i, (_, r) in enumerate(sub.iterrows()):
        f = lambda v: f"${v:+.3f}$".replace("0.", ".")
        lines.append(" & ".join([NAMES[c] if i == 0 else "", LRS[r.lr], str(int(r.n)), f"{r.gap:.2f}",
                                 f(r.d_shift_coco_order), f(r.d_orth_coco_order), f(r.d_shift_vg_relation),
                                 f(r.d_shift_vg_attribution), f(r.d_shift_coco_test)]) + r" \\")
    lines.append(r"\midrule")
open("final_submission/report/lr_table.tex", "w").write("\n".join(lines[:-1]) + "\n")
print("wrote lr_table.tex")
