"""Modality-gap measurement (Approach 01) and post-hoc gap shifts (Approach 02).

All inputs are L2-normalised embeddings: img (n_img, d), txt (n_txt, d).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .common import l2n


# --- Approach 01: measure ------------------------------------------------

def gap_vector(img, txt, stat="mean"):
    """Delta = centre(img) - centre(txt) (Liang et al. 2022, Sec 4.2).

    stat="median" uses the coordinate-wise median, which is less sensitive
    to outlier captions.
    """
    f = np.mean if stat == "mean" else np.median
    return f(img, axis=0) - f(txt, axis=0)


def remaining_gap(img, txt):
    return float(np.linalg.norm(img.mean(0) - txt.mean(0)))


def _uniformity(x, n=5000, seed=0):
    """Wang & Isola (2020): log E exp(-2 ||u - v||^2) over random pairs."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(x), n), rng.integers(0, len(x), n)
    keep = i != j
    d2 = ((x[i[keep]] - x[j[keep]]) ** 2).sum(1)
    return float(np.log(np.exp(-2 * d2).mean()))


def _var_frac_along(x, u):
    """Fraction of x's within-modality variance that lies along unit vector u.

    A random direction gets 1/d (~0.002 for d=512). Close to that value
    means the gap direction is (nearly) orthogonal to how the embeddings
    vary, i.e. the Chowers et al. condition.
    """
    xc = x - x.mean(0)
    return float((xc @ u).var() / xc.var(0).sum())


def gap_metrics(img, txt, txt2img, seed=0):
    """All Approach-01 numbers for one (model, dataset). txt2img maps caption -> image."""
    delta = gap_vector(img, txt)
    u = delta / np.linalg.norm(delta)
    mi, mt = img.mean(0), txt.mean(0)
    rng = np.random.default_rng(seed)

    matched = (img[txt2img] * txt).sum(1)
    rand_img = rng.integers(0, len(img), len(txt))
    unmatched = (img[rand_img] * txt).sum(1)[rand_img != txt2img]

    # can a linear classifier tell image from text embeddings?
    X = np.concatenate([img, txt])
    y = np.r_[np.zeros(len(img)), np.ones(len(txt))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sep_acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)

    return dict(
        gap_norm=float(np.linalg.norm(delta)),
        gap_norm_median=float(np.linalg.norm(gap_vector(img, txt, "median"))),
        centroid_cos=float(mi @ mt / np.linalg.norm(mi) / np.linalg.norm(mt)),
        img_centroid_norm=float(np.linalg.norm(mi)),  # near 1 = narrow cone
        txt_centroid_norm=float(np.linalg.norm(mt)),
        proj_img_mean=float((img @ u).mean()), proj_img_std=float((img @ u).std()),
        proj_txt_mean=float((txt @ u).mean()), proj_txt_std=float((txt @ u).std()),
        var_frac_img=_var_frac_along(img, u), var_frac_txt=_var_frac_along(txt, u),
        var_frac_random=1.0 / img.shape[1],
        matched_cos=float(matched.mean()), unmatched_cos=float(unmatched.mean()),
        alignment=float(((img[txt2img] - txt) ** 2).sum(1).mean()),
        uniformity_img=_uniformity(img, seed=seed), uniformity_txt=_uniformity(txt, seed=seed),
        linear_sep_acc=float(sep_acc),
    )


# --- Approach 02: alter ----------------------------------------------------

def apply_shift(img, txt, delta, alpha, mode="text"):
    """Move embeddings along the gap vector, then re-normalise to the sphere.

    mode="text": y' = N(y + alpha * Delta), images untouched
                 (shift one modality toward the other, as in Chowers et al.)
    mode="both": x' = N(x - alpha/2 * Delta), y' = N(y + alpha/2 * Delta)
                 (symmetric version of Liang et al.'s shift experiment)
    alpha = 1 closes the gap before re-normalisation; alpha < 0 widens it.
    """
    if mode == "text":
        return img, l2n(txt + alpha * delta)
    if mode == "both":
        return l2n(img - alpha / 2 * delta), l2n(txt + alpha / 2 * delta)
    raise ValueError(mode)


def alpha_for_target_gap(img, txt, delta, target, mode="text", alpha_max=2.0, iters=40):
    """Find alpha >= 0 whose shift leaves `target` remaining gap (mean/median margin shift).

    The remaining gap decreases with alpha up to some alpha*, then grows
    again (overshoot), so we bisect on [0, alpha*].
    """
    grid = np.linspace(0, alpha_max, 81)
    gaps = [remaining_gap(*apply_shift(img, txt, delta, a, mode)) for a in grid]
    hi = grid[int(np.argmin(gaps))]
    if target >= gaps[0]:
        return 0.0
    if target <= min(gaps):
        return float(hi)
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if remaining_gap(*apply_shift(img, txt, delta, mid, mode)) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
