"""Extra analyses on cached embeddings for one model (final-report additions).

X2  direction-removal control: remove each caption's deviation along the gap
    direction u, along random unit directions, and along top text principal
    components; compare the accuracy change.
X3  one-coordinate probe: how well does d = y.u alone separate true captions
    from their hard negatives?
X4  caption length: is y.u correlated with caption length (words)?
X5  deviation-preserving gap closing: close the gap with the text shift, then
    put each caption's own deviation along u back (rescaled like the rest of
    the vector), and evaluate clean and under embedding noise.

Usage: python scripts/extra_experiments.py --model negclip --out results/extra/negclip.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.data import BENCHMARKS  # noqa: E402
from src.embed import get_embeddings  # noqa: E402
from src.evaluate import add_noise, eval_cases, eval_retrieval  # noqa: E402
from src.gap import alpha_for_target_gap, apply_shift, gap_vector, remaining_gap  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n-random", type=int, default=20)
a = p.parse_args()

CASES = [("coco_order", "acc"), ("vg_relation", "macro_acc"), ("vg_attribution", "macro_acc"), ("sugarcrepe", "acc")]


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def remove_subspace(txt, V):
    """Remove each text's deviation (from the benchmark mean) inside span(V); V is (k, d) orthonormal."""
    c = txt @ V.T
    return l2n(txt - (c - c.mean(0)) @ V)


fit = BENCHMARKS["coco_val"]()
fit_img, fit_txt, _ = get_embeddings(a.model, fit)
delta = gap_vector(fit_img, fit_txt)
u = delta / np.linalg.norm(delta)
benches = {name: BENCHMARKS[name]() for name, _ in CASES}
embs = {name: get_embeddings(a.model, b)[:2] for name, b in benches.items()}
res = dict(model=a.model, gap_val=float(np.linalg.norm(delta)))

# ---- X2: direction-removal control ------------------------------------------------
rng = np.random.default_rng(0)
rand = l2n(rng.standard_normal((a.n_random, fit_txt.shape[1])).astype(np.float32))
tc = fit_txt - fit_txt.mean(0)
_, _, vt = np.linalg.svd(tc, full_matrices=False)
pcs = vt[:20].astype(np.float32)
var_u = float(((tc @ u) ** 2).mean() / (tc ** 2).sum(1).mean())
x2 = {}
for name, metric in CASES:
    img, txt = embs[name]
    base = eval_cases(benches[name], img, txt)[metric]
    row = dict(base=base, u=eval_cases(benches[name], img, remove_subspace(txt, u[None]))[metric] - base)
    r = [eval_cases(benches[name], img, remove_subspace(txt, v[None]))[metric] - base for v in rand]
    row.update(random_mean=float(np.mean(r)), random_min=float(np.min(r)), random_max=float(np.max(r)))
    for k in (1, 5, 20):
        row[f"pc{k}"] = eval_cases(benches[name], img, remove_subspace(txt, pcs[:k]))[metric] - base
    x2[name] = row
res["x2_removal"] = x2
res["var_frac_txt_u"] = var_u
res["var_frac_txt_pc1"] = float(((tc @ pcs[0]) ** 2).mean() / (tc ** 2).sum(1).mean())

# ---- X3: one-coordinate probe -------------------------------------------------------
x3 = {}
for name, _ in CASES:
    b = benches[name]
    img, txt = embs[name]
    d = txt @ u
    D = d[np.array(b["case_texts"])]          # (N, K)
    c = np.asarray(b["correct"])
    true = D[np.arange(len(D)), c]
    neg = np.delete(D, c, axis=1) if np.ndim(c) == 0 else np.stack([np.delete(r, ci) for r, ci in zip(D, c)])
    x3[name] = dict(true_is_max_d=float((D.argmax(1) == c).mean()), true_is_min_d=float((D.argmin(1) == c).mean()),
                    auc_true_gt_neg=float((true[:, None] > neg).mean()), mean_d_true_minus_neg=float((true[:, None] - neg).mean()),
                    chance=float(1 / D.shape[1]))
    if b.get("groups") is not None and name == "sugarcrepe":
        groups = np.array(b["groups"])
        x3[name]["by_subset"] = {g: dict(auc_true_gt_neg=float((true[groups == g, None] > neg[groups == g]).mean()),
                                         mean_d_true_minus_neg=float((true[groups == g, None] - neg[groups == g]).mean()))
                                 for g in np.unique(groups)}
res["x3_probe"] = x3

# ---- X4: caption length ----------------------------------------------------------------
x4 = {}
for name, _ in CASES:
    b = benches[name]
    img, txt = embs[name]
    d = txt @ u
    L = np.array([len(t.split()) for t in b["texts"]])
    rho = spearmanr(d, L)
    x4[name] = dict(rho_d_length=float(rho.correlation), p=float(rho.pvalue), n=int(len(L)))
    if name == "sugarcrepe":
        CT = np.array(b["case_texts"])
        c = np.asarray(b["correct"])
        groups = np.array(b["groups"])
        pos, negi = CT[np.arange(len(CT)), c], CT[np.arange(len(CT)), 1 - c]
        dl, dd = L[negi] - L[pos], d[negi] - d[pos]
        x4[name]["by_subset"] = {}
        for g in np.unique(groups):
            m = groups == g
            r = spearmanr(dd[m], dl[m])
            x4[name]["by_subset"][g] = dict(mean_len_neg_minus_pos=float(dl[m].mean()), mean_d_neg_minus_pos=float(dd[m].mean()),
                                            rho_dd_dl=float(r.correlation), frac_neg_lower_d=float((dd[m] < 0).mean()))
res["x4_length"] = x4

# ---- X5: deviation-preserving gap closing -------------------------------------------------
def shift_keep_dev(txt, alpha):
    ys = l2n(txt + alpha * delta)
    p0, ps = txt @ u, ys @ u
    s = 1.0 / np.linalg.norm(txt + alpha * delta, axis=1)       # perpendicular parts scale by this
    target = ps.mean() + (p0 - p0.mean()) * s
    return l2n(ys + np.outer(target - ps, u))


alpha = alpha_for_target_gap(fit_img, fit_txt, delta, 0.1, mode="text")
res["x5_alpha"] = float(alpha)
res["x5_gap_after"] = dict(text=float(remaining_gap(*apply_shift(fit_img, fit_txt, delta, alpha, "text"))),
                           keep_dev=float(remaining_gap(fit_img, shift_keep_dev(fit_txt, alpha))))
ret_b = BENCHMARKS["coco_test"]()
ret_e = get_embeddings(a.model, ret_b)[:2]
x5 = []
for sigma in (0.0, 0.5, 1.0):
    for variant in ("none", "text", "keep_dev"):
        seeds = [0] if sigma == 0 else [0, 1, 2]
        acc = {}
        for name, metric in CASES + [("coco_test", "i2t_R@1"), ("coco_test", "t2i_R@1")]:
            img, txt = ret_e if name == "coco_test" else embs[name]
            if variant == "text":
                txt = apply_shift(img, txt, delta, alpha, "text")[1]
            elif variant == "keep_dev":
                txt = shift_keep_dev(txt, alpha)
            vals = []
            for sd in seeds:
                r = np.random.default_rng(1000 + sd)
                xi, yt = add_noise(img, sigma, r), add_noise(txt, sigma, r)
                if name == "coco_test":
                    vals.append(eval_retrieval(ret_b, xi, yt, ks=(1,))[metric])
                else:
                    vals.append(eval_cases(benches[name], xi, yt)[metric])
            acc[f"{name}:{metric}"] = float(np.mean(vals))
        x5.append(dict(sigma=sigma, variant=variant, **acc))
res["x5_keep_dev"] = x5

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
print("wrote", a.out)
