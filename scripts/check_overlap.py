"""Check that our held-out COCO images were never seen by NegCLIP in training.

NegCLIP's training file (data/negclip/train_neg_clip.tsv, from the ARO repo)
contains COCO *val2014* images, i.e. it used Karpathy train + restval. Our
gap measurement and evaluation use Karpathy val / test, which must not overlap.

Usage: .venv/bin/python scripts/check_overlap.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import COCO_DIR, DATA  # noqa: E402

neg = {}
for split in ("train", "valid"):
    df = pd.read_csv(os.path.join(DATA, "negclip", f"{split}_neg_clip.tsv"), sep="\t")
    neg[split] = set(df["filepath"].map(os.path.basename))
    print(f"NegCLIP {split}: {len(df)} rows, {len(neg[split])} unique images")

ok = True
for split in ("train", "val", "test"):
    with open(os.path.join(COCO_DIR, f"coco_karpathy_{split}.json")) as f:
        kp = {os.path.basename(x["image"]) for x in json.load(f)}
    for nsplit, imgs in neg.items():
        n = len(kp & imgs)
        print(f"Karpathy {split:5s} ({len(kp)} images) & NegCLIP {nsplit}: {n} shared")
        if split in ("val", "test") and nsplit == "train" and n > 0:
            ok = False

print("OK: Karpathy val/test are unseen by NegCLIP training" if ok else "WARNING: overlap with NegCLIP training data")
