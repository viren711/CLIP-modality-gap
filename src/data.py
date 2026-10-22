"""Benchmarks, all expressed in one format so embeddings can be cached and reused.

A benchmark is a dict:
    images      list of (path, bbox or None)   - unique images / crops to encode
    texts       list of str                    - unique captions to encode
    case_image  int array (N,)                 - image index of each test case
    case_texts  int array (N, K)               - caption options of each test case
    correct     int                            - column of the correct caption
    groups      list of str (N,) or None       - relation / attribute label (for macro acc)
COCO retrieval benchmarks instead carry `txt2img` (caption -> its image).

Every shift experiment then works on the cached (image, text) embedding
matrices plus these index arrays, with no re-encoding.
"""
import json
import os
import random

import numpy as np

from .common import ARO_DIR, COCO_DIR, pre_caption


class _Interner:
    """Assigns a stable index to each unique item."""

    def __init__(self):
        self.items, self.index = [], {}

    def __call__(self, item):
        if item not in self.index:
            self.index[item] = len(self.items)
            self.items.append(item)
        return self.index[item]


def coco_karpathy(split):
    """COCO Karpathy val/test: 5k images x ~5 captions, for gap + retrieval."""
    with open(os.path.join(COCO_DIR, f"coco_karpathy_{split}.json")) as f:
        ann = json.load(f)
    images, texts, txt2img = [], [], []
    for i, item in enumerate(ann):
        images.append((os.path.join(COCO_DIR, item["image"]), None))
        for cap in item["caption"]:
            texts.append(pre_caption(cap))
            txt2img.append(i)
    return dict(name=f"coco_{split}", images=images, texts=texts, txt2img=np.array(txt2img))


def _vg(kind):
    """ARO VG-Relation / VG-Attribution: one image crop, [false, true] caption."""
    fname = {"relation": "visual_genome_relation.json", "attribution": "visual_genome_attribution.json"}[kind]
    with open(os.path.join(ARO_DIR, fname)) as f:
        ann = json.load(f)
    imgs, txts = _Interner(), _Interner()
    case_image, case_texts, groups = [], [], []
    for item in ann:
        bbox = (item["bbox_x"], item["bbox_y"], item["bbox_x"] + item["bbox_w"], item["bbox_y"] + item["bbox_h"])
        case_image.append(imgs((os.path.join(ARO_DIR, "images", item["image_path"]), bbox)))
        case_texts.append([txts(item["false_caption"]), txts(item["true_caption"])])
        if kind == "relation":
            groups.append(item["relation_name"])
        else:
            groups.append(f"{item['attributes'][0]}_{item['attributes'][1]}")
    return dict(name=f"vg_{kind}", images=imgs.items, texts=txts.items,
                case_image=np.array(case_image), case_texts=np.array(case_texts),
                correct=1, groups=groups)


def vg_relation():
    return _vg("relation")


def vg_attribution():
    return _vg("attribution")


# --- COCO-Order -----------------------------------------------------------
# Same four perturbations as the ARO repo (dataset_zoo/perturbations.py).
# The ARO repo draws them unseeded on every load; we draw them once with a
# fixed seed and cache them, so every model sees the same negatives.

def _trigrams(tokens):
    return [tokens[i:i + 3] for i in range(0, len(tokens), 3)]


def _make_order_negatives(caption, nlp, nltk):
    doc = nlp(caption)
    tokens = np.array([t.text for t in doc])
    noun = [i for i, t in enumerate(doc) if t.tag_ in ("NN", "NNS", "NNP", "NNPS")]
    adj = [i for i, t in enumerate(doc) if t.tag_ in ("JJ", "JJR", "JJS")]

    a = tokens.copy()  # shuffle nouns and adjectives (among themselves)
    a[noun] = np.random.permutation(a[noun])
    a[adj] = np.random.permutation(a[adj])

    b = tokens.copy()  # shuffle everything except nouns and adjectives
    rest = np.ones(len(b), dtype=bool)
    rest[noun + adj] = False
    b[rest] = np.random.permutation(b[rest])

    words = nltk.word_tokenize(caption)
    c = _trigrams(words)  # shuffle within trigrams
    for tri in c:
        random.shuffle(tri)
    d = _trigrams(words)  # shuffle the trigrams
    random.shuffle(d)

    return [" ".join(a), " ".join(b), " ".join(w for t in c for w in t), " ".join(w for t in d for w in t)]


def coco_order(split="test", seed=0):
    """ARO COCO-Order: each caption vs 4 word-order perturbations, correct = 0."""
    cache = os.path.join(ARO_DIR, f"coco_order_{split}_seed{seed}.json")
    if not os.path.exists(cache):
        import nltk
        import spacy
        nlp = spacy.load("en_core_web_sm")
        random.seed(seed)
        np.random.seed(seed)
        with open(os.path.join(COCO_DIR, f"coco_karpathy_{split}.json")) as f:
            ann = json.load(f)
        cases = []
        for item in ann:
            for cap in item["caption"]:
                options = [cap] + _make_order_negatives(cap, nlp, nltk)
                cases.append(dict(image=item["image"], captions=[pre_caption(o) for o in options]))
        os.makedirs(ARO_DIR, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(cases, f)
    with open(cache) as f:
        cases = json.load(f)

    imgs, txts = _Interner(), _Interner()
    case_image = [imgs((os.path.join(COCO_DIR, c["image"]), None)) for c in cases]
    case_texts = [[txts(t) for t in c["captions"]] for c in cases]
    return dict(name=f"coco_order_{split}", images=imgs.items, texts=txts.items,
                case_image=np.array(case_image), case_texts=np.array(case_texts),
                correct=0, groups=None)


BENCHMARKS = {
    "coco_val": lambda: coco_karpathy("val"),
    "coco_test": lambda: coco_karpathy("test"),
    "vg_relation": vg_relation,
    "vg_attribution": vg_attribution,
    "coco_order": coco_order,
}
