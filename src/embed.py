"""Encode a benchmark's images and texts once per model and cache to disk.

Cache file: results/embeddings/<model>/<benchmark>.npz with
    img (n_images, 512) and txt (n_texts, 512), both L2-normalised float32,
    plus logit_scale and a fingerprint of the image/text lists.
"""
import hashlib
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import EMB_DIR, l2n


class _Images(Dataset):
    def __init__(self, specs, preprocess):
        self.specs, self.preprocess = specs, preprocess

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        path, bbox = self.specs[i]
        img = Image.open(path).convert("RGB")
        if bbox is not None:
            img = img.crop(bbox)  # ARO evaluates on the relation/attribute crop
        return self.preprocess(img)


@torch.no_grad()
def encode_images(model, preprocess, specs, batch_size=128, workers=4):
    loader = DataLoader(_Images(specs, preprocess), batch_size=batch_size, num_workers=workers)
    out = [model.encode_image(x).float().numpy() for x in tqdm(loader, desc="images", mininterval=30)]
    return l2n(np.concatenate(out))


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=512):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="texts", mininterval=30):
        out.append(model.encode_text(tokenizer(texts[i:i + batch_size])).float().numpy())
    return l2n(np.concatenate(out))


def fingerprint(bench):
    h = hashlib.sha1()
    for path, bbox in bench["images"]:
        h.update(f"{os.path.basename(path)}{bbox}".encode())
    for t in bench["texts"]:
        h.update(t.encode())
    return h.hexdigest()


def cache_path(model_name, bench_name):
    return os.path.join(EMB_DIR, model_name, f"{bench_name}.npz")


def get_embeddings(model_name, bench, model=None, preprocess=None, tokenizer=None):
    """Load cached embeddings, or compute them (model/preprocess/tokenizer required)."""
    path = cache_path(model_name, bench["name"])
    fp = fingerprint(bench)
    if os.path.exists(path):
        z = np.load(path)
        if str(z["fingerprint"]) == fp:
            return z["img"], z["txt"], float(z["logit_scale"])
        print(f"{path} is stale (benchmark changed), recomputing")
    if model is None:
        raise FileNotFoundError(f"No cached embeddings at {path}; run scripts/embed_all.py first")
    img = encode_images(model, preprocess, bench["images"])
    txt = encode_texts(model, tokenizer, bench["texts"])
    scale = float(model.logit_scale.exp())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, img=img, txt=txt, logit_scale=scale, fingerprint=fp)
    return img, txt, scale
