"""Encode every benchmark with every model and cache the embeddings.

Usage:
    .venv/bin/python scripts/embed_all.py --models clip negclip \
        --benchmarks coco_val coco_test coco_order vg_relation vg_attribution
Already-cached (model, benchmark) pairs are skipped.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.data import BENCHMARKS  # noqa: E402
from src.embed import cache_path, get_embeddings  # noqa: E402
from src.models import load_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["clip", "negclip"])
parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
parser.add_argument("--threads", type=int, default=6)
args = parser.parse_args()

torch.set_num_threads(args.threads)
for m in args.models:
    model, preprocess, tokenizer = load_model(m)
    for b in args.benchmarks:
        bench = BENCHMARKS[b]()
        print(f"[{m} / {b}] {len(bench['images'])} images, {len(bench['texts'])} texts", flush=True)
        t = time.time()
        img, txt, scale = get_embeddings(m, bench, model, preprocess, tokenizer)
        print(f"[{m} / {b}] done in {time.time() - t:.0f}s -> {cache_path(m, b)} (logit scale {scale:.1f})", flush=True)
