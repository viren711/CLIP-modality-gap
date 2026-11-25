"""Does the gap direction encode how 'natural' a caption reads?

For every COCO-Order caption (true + 4 word-shuffled negatives) we compute:
  - its projection on the gap direction u = Delta/||Delta|| (from cached
    text embeddings; higher = further toward the text side)
  - its GPT-2 perplexity (higher = less natural / more surprising a string)
and report the Spearman correlation between the two, per model. A positive
correlation supports the Section 5.7 finding that shuffled captions (higher
perplexity) sit further on the text side, which is what the gap-closing
shift rewards.

Usage: .venv/bin/python scripts/naturalness.py --models clip negclip ft_negclip
Writes results/naturalness.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

from src.common import RESULTS  # noqa: E402
from src.data import BENCHMARKS  # noqa: E402
from src.embed import get_embeddings  # noqa: E402
from src.gap import gap_vector  # noqa: E402
from src.models import default_device  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["clip", "negclip"])
parser.add_argument("--device", default=default_device())
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lm", default="gpt2", help="HF id or local folder of the GPT-2 weights")
parser.add_argument("--out", default=os.path.join(RESULTS, "naturalness.csv"))
args = parser.parse_args()

device = args.device
tok = GPT2TokenizerFast.from_pretrained(args.lm)
tok.pad_token = tok.eos_token
gpt2 = GPT2LMHeadModel.from_pretrained(args.lm).to(device).eval()


@torch.no_grad()
def perplexities(texts, batch_size):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
        logits = gpt2(**enc).logits[:, :-1]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100
        ).reshape(labels.size(0), -1)
        n_tok = (labels[:, 1:] != -100).sum(1).clamp(min=1)
        out.extend((loss.sum(1) / n_tok).exp().cpu().tolist())
    return out


order = BENCHMARKS["coco_order"]()
ppl = np.array(perplexities(order["texts"], args.batch_size))
val = BENCHMARKS["coco_val"]()
rows = []
for m in args.models:
    fit_img, fit_txt, _ = get_embeddings(m, val)
    delta = gap_vector(fit_img, fit_txt)
    u = delta / np.linalg.norm(delta)
    _, txt, _ = get_embeddings(m, order)
    proj = txt @ u
    rho, p = spearmanr(proj, ppl)
    rows.append(dict(model=m, spearman_rho=rho, p_value=p, n=len(proj)))
    print(f"{m:12s}  rho(projection on gap dir, GPT-2 perplexity) = {rho:+.3f}  (p={p:.2e}, n={len(proj)})", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"wrote {args.out}")
