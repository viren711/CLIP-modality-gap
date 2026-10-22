"""Model registry. Every model is an OpenCLIP ViT-B/32 so results are comparable.

clip     : OpenAI CLIP ViT-B/32 weights (the starting point of NegCLIP)
negclip  : NegCLIP, COCO fine-tuned WITH hard negatives (Yuksekgonul et al. 2023)
clipft   : control, COCO fine-tuned WITHOUT hard negatives (to be trained,
           checkpoint expected at checkpoints/clipft.pth)

The OpenAI weights were trained with QuickGELU, and NegCLIP was fine-tuned
from them, so we build the "-quickgelu" architecture for every checkpoint.
(The ARO repo builds plain "ViT-B-32" for NegCLIP, i.e. with standard GELU;
use arch="ViT-B-32" to reproduce that exactly.)
"""
import os

import open_clip
import torch

from .common import CKPT_DIR

MODELS = {
    "clip": dict(arch="ViT-B-32-quickgelu", pretrained="openai"),
    "negclip": dict(arch="ViT-B-32-quickgelu", pretrained=os.path.join(CKPT_DIR, "negclip.pth")),
    "clipft": dict(arch="ViT-B-32-quickgelu", pretrained=os.path.join(CKPT_DIR, "clipft.pth")),
}


def load_model(name, device="cpu", arch=None):
    cfg = MODELS[name]
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch or cfg["arch"], pretrained=cfg["pretrained"], device=device)
    tokenizer = open_clip.get_tokenizer(arch or cfg["arch"])
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def logit_scale(model):
    return float(model.logit_scale.exp())
