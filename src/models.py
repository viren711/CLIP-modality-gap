"""Model registry. Every model is an OpenCLIP ViT-B/32 so results are comparable.

clip     : OpenAI CLIP ViT-B/32 weights (the starting point of NegCLIP)
negclip  : released NegCLIP, COCO fine-tuned WITH hard negatives (Yuksekgonul et al. 2023)
<name>   : any checkpoint we fine-tuned ourselves, saved by scripts/train_finetune.py
           as checkpoints/<name>.pt, e.g. the 2x2 control grid
             ft_plain   - no hard negatives            (the CLIP-FT control)
             ft_hardimg - hard negative images only
             ft_hardcap - hard negative captions only
             ft_negclip - both (our NegCLIP reproduction)

The OpenAI weights were trained with QuickGELU, and NegCLIP was fine-tuned
from them, so we build the "-quickgelu" architecture for every checkpoint.
negclip_gelu (standard GELU) reproduces how the ARO repo evaluates NegCLIP.
"""
import os

import open_clip
import torch

from .common import CKPT_DIR

ARCH = "ViT-B-32-quickgelu"
_NEGCLIP = os.path.join(CKPT_DIR, "negclip_state_dict.pt")  # made by scripts/convert_negclip.py
MODELS = {  # name -> (pretrained, arch)
    "clip": ("openai", ARCH),
    "negclip": (_NEGCLIP, ARCH),
    # Same weights with standard GELU, i.e. how the ARO repo evaluates NegCLIP.
    # NegCLIP was TRAINED with QuickGELU (the fork builds OpenAI models with
    # quick_gelu=True), so this is a mismatch, but it is what reproduces the
    # paper's numbers (COCO-Order 0.87 vs 0.82 with QuickGELU). Footnote only.
    "negclip_gelu": (_NEGCLIP, "ViT-B-32"),
}


def checkpoint_for(name):
    """(pretrained, arch) for a registry name or a fine-tuned checkpoints/<name>.pt."""
    if name in MODELS:
        return MODELS[name]
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown model '{name}': no entry in MODELS and no {path}")
    return path, ARCH


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name, device="cpu", arch=None):
    pretrained, default_arch = checkpoint_for(name)
    arch = arch or default_arch
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def logit_scale(model):
    return float(model.logit_scale.exp())
