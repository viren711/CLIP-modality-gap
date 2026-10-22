"""Metrics computed from (possibly shifted / noised) embeddings."""
import numpy as np


def case_scores(bench, img, txt):
    """(N, K) cosine similarity of each test case's image with its caption options."""
    return np.einsum("nd,nkd->nk", img[bench["case_image"]], txt[bench["case_texts"]])


def eval_cases(bench, img, txt):
    """ARO-style accuracy: is the correct caption scored highest?

    macro_acc follows the ARO paper (mean over relations / attribute pairs;
    for VG-Attribution only pairs with >= 25 cases, as in the ARO repo).
    margin = score(correct) - best score among the wrong captions.
    """
    s = case_scores(bench, img, txt)
    c = bench["correct"]
    correct = s.argmax(1) == c
    others = np.delete(s, c, axis=1)
    out = dict(acc=float(correct.mean()), margin=float((s[:, c] - others.max(1)).mean()))
    if bench["groups"] is not None:
        groups = np.array(bench["groups"])
        min_count = 25 if bench["name"] == "vg_attribution" else 1
        accs = [correct[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() >= min_count]
        out["macro_acc"] = float(np.mean(accs))
    return out


def eval_retrieval(bench, img, txt, ks=(1, 5, 10), chunk=1000):
    """COCO 5k image<->text Recall@K (each image has ~5 captions)."""
    t2i = bench["txt2img"]
    # text -> image: rank of the caption's own image
    t2i_rank = np.empty(len(txt), dtype=np.int64)
    for s in range(0, len(txt), chunk):
        sims = txt[s:s + chunk] @ img.T
        own = sims[np.arange(len(sims)), t2i[s:s + chunk]]
        t2i_rank[s:s + chunk] = (sims > own[:, None]).sum(1)
    # image -> text: rank of the best of the image's own captions
    i2t_rank = np.empty(len(img), dtype=np.int64)
    for s in range(0, len(img), chunk):
        sims = img[s:s + chunk] @ txt.T
        mask = t2i[None, :] == np.arange(s, s + len(sims))[:, None]
        best_own = np.where(mask, sims, -np.inf).max(1)
        i2t_rank[s:s + chunk] = (np.where(mask, -np.inf, sims) > best_own[:, None]).sum(1)
    out = {}
    for k in ks:
        out[f"i2t_R@{k}"] = float((i2t_rank < k).mean())
        out[f"t2i_R@{k}"] = float((t2i_rank < k).mean())
    return out


def add_noise(x, sigma, rng):
    """Gaussian perturbation with expected norm ~sigma, then back onto the sphere."""
    if sigma == 0:
        return x
    noise = rng.standard_normal(x.shape).astype(x.dtype) * sigma / np.sqrt(x.shape[1])
    noisy = x + noise
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
