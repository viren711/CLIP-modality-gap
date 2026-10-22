"""Paths and small helpers shared by all scripts."""
import os
import random
import re

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
COCO_DIR = os.path.join(DATA, "coco")
ARO_DIR = os.path.join(DATA, "aro")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")
EMB_DIR = os.path.join(RESULTS, "embeddings")
FIG_DIR = os.path.join(RESULTS, "figures")


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pre_caption(caption, max_words=30):
    """Caption cleaning used by BLIP / the NegCLIP (ARO) repo for COCO.

    Lower-case, drop punctuation, collapse whitespace, truncate to max_words.
    Kept identical so COCO-Order numbers are comparable with the paper.
    """
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.rstrip("\n").strip(" ")
    words = caption.split(" ")
    if len(words) > max_words:
        caption = " ".join(words[:max_words])
    return caption


def l2n(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)
