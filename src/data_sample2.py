"""Load the local image/caption pairs in the existing embedding benchmark format.

Each JSONL record contributes (image_0, caption_0), then (image_1, caption_1).
Image and text row i are a positive pair; captions are kept verbatim.
"""
import json
import os

import numpy as np

from .common import DATA


def sample2(data_dir=DATA):
    """Return paired images/texts, an identity txt2img array, and row metadata.

    Preserve JSONL record order and, within each record, pair order 0 then 1.
    For the supplied 400 records this produces 800 image/text pairs.
    """
    annotation_path = os.path.join(data_dir, "text", "examples.jsonl")
    images, texts, rows = [], [], []
    seen_ids = set()
    with open(annotation_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = ("id", "image_0", "image_1", "caption_0", "caption_1")
            missing = [key for key in required if key not in item]
            if missing:
                raise ValueError(f"{annotation_path}:{line_number}: missing fields {missing}")
            example_id = item["id"]
            if example_id in seen_ids:
                raise ValueError(f"{annotation_path}:{line_number}: duplicate id {example_id}")
            seen_ids.add(example_id)
            for pair_index in (0, 1):
                image_name = item[f"image_{pair_index}"]
                caption = item[f"caption_{pair_index}"]
                if not isinstance(image_name, str) or not image_name:
                    raise ValueError(f"Example {example_id}: invalid image_{pair_index}")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"Example {example_id}: empty or invalid caption_{pair_index}")
                image_file = image_name + ".png"
                image_path = os.path.join(data_dir, "images", image_file)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Example {example_id}: missing image {image_path}")
                images.append((image_path, None))
                texts.append(caption)
                rows.append(dict(row=len(rows), id=example_id, pair_index=pair_index,
                                 image=f"images/{image_file}", caption=caption,
                                 **{key: item[key] for key in
                                    ("tag", "secondary_tag", "num_main_preds", "collapsed_tag")
                                    if key in item}))
    if not rows:
        raise ValueError(f"No image/caption pairs found in {annotation_path}")
    return dict(name="sample2_positive", images=images, texts=texts,
                txt2img=np.arange(len(texts), dtype=np.int64), rows=rows)
