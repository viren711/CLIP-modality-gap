"""Fine-tune OpenAI CLIP ViT-B/32 on COCO with or without NegCLIP hard negatives.

Re-implements the NegCLIP recipe (github.com/vinid/neg_clip, a fork of
open_clip) in one file so the control and NegCLIP differ ONLY in the flags:

  --hard-images    add, for each image, one of its 3 nearest-neighbour COCO
                   images (with its own caption) to the batch
  --hard-captions  add, for each caption, one word-swapped negative caption
                   (scored image->text only; it has no matching image)

  flags          model name     role
  (none)         ft_plain       CLIP-FT control: COCO fine-tuning only
  --hard-images  ft_hardimg     batch composition only
  --hard-captions ft_hardcap    hard negative captions only
  both           ft_negclip     NegCLIP reproduction

Recipe (from the fork / released checkpoint "mscoco_all_neg_all_bs256_1e-6"):
batch 256, 5 epochs, AdamW (betas 0.9/0.98, eps 1e-6, wd 0.2, no wd on
gains/biases/logit scale), cosine LR with 50 warmup steps, fp16 AMP,
learnable logit scale clamped to 100, RandomResizedCrop(scale 0.9-1.0).
Batch layout and loss match the fork exactly:
  images = [I, I_hard]            texts = [T, T(I_hard), T_neg, T_neg(I_hard)]
  loss   = (CE(images x all texts) + CE(first len(images) texts x images)) / 2

After every epoch we report COCO-val retrieval and the modality gap, so the
gap's evolution during fine-tuning is logged too (log.jsonl).

Usage:
  python scripts/train_finetune.py --name ft_negclip --hard-images --hard-captions --lr 1e-6
"""
import argparse
import ast
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.common import DATA, ROOT  # noqa: E402
from src.models import ARCH  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--hard-images", action="store_true")
parser.add_argument("--hard-captions", action="store_true")
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--wd", type=float, default=0.2)
parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train-tsv", default=os.path.join(DATA, "negclip", "train_neg_clip.tsv"))
parser.add_argument("--val-tsv", default=os.path.join(DATA, "negclip", "valid_neg_clip.tsv"))
parser.add_argument("--image-root", default=DATA, help="tsv file paths are relative to this (coco/train2014/...)")
parser.add_argument("--out-dir", default=None, help="default: runs/<name>_lr<lr>_s<seed>")
parser.add_argument("--save-every-epoch", action="store_true")
parser.add_argument("--grad-checkpointing", action="store_true", help="lower memory, slower")
parser.add_argument("--max-steps", type=int, default=None, help="debug: stop after this many steps")
parser.add_argument("--device", default="cuda", help="cpu only for debugging (no AMP)")
parser.add_argument("--val-limit", type=int, default=None, help="debug: validate on the first N pairs")
args = parser.parse_args()

out_dir = args.out_dir or os.path.join(ROOT, "runs", f"{args.name}_lr{args.lr:g}_s{args.seed}")
os.makedirs(out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = args.device
cuda = device.startswith("cuda")


class NegClipTsv(Dataset):
    def __init__(self, path, transform, tokenizer, hard_images, hard_captions):
        df = pd.read_csv(path, sep="\t", converters={"neg_caption": ast.literal_eval, "neg_image": ast.literal_eval})
        self.images = [os.path.join(args.image_root, p) for p in df["filepath"]]
        self.captions = df["title"].astype(str).tolist()
        self.neg_captions = df["neg_caption"].tolist()
        self.neg_images = df["neg_image"].tolist()
        self.transform, self.tok = transform, tokenizer
        self.hard_images, self.hard_captions = hard_images, hard_captions

    def __len__(self):
        return len(self.captions)

    def _item(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        text = self.tok([self.captions[idx]])[0]
        neg = self.tok([str(random.choice(self.neg_captions[idx]))])[0] if self.hard_captions else text
        return image, text, neg

    def __getitem__(self, idx):
        image, text, neg = self._item(idx)
        if self.hard_images:
            h_image, h_text, h_neg = self._item(random.choice(self.neg_images[idx]))
        else:  # placeholders, never used
            h_image, h_text, h_neg = torch.zeros(0), text, neg
        return image, text, neg, h_image, h_text, h_neg


def loss_fn(img_f, txt_f, scale):
    """NegCLIP loss: extra text rows (hard captions) only act as image->text negatives."""
    labels = torch.arange(len(img_f), device=img_f.device)
    logits_i = scale * img_f @ txt_f.T                 # (n_img, n_txt)
    logits_t = scale * txt_f[:len(img_f)] @ img_f.T    # (n_img, n_img)
    return (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)) / 2


@torch.no_grad()
def validate(model, preprocess, tokenizer):
    """COCO-val retrieval (1 caption per image, as in the fork) + modality gap."""
    model.eval()
    df = pd.read_csv(args.val_tsv, sep="\t", nrows=args.val_limit)
    paths = [os.path.join(args.image_root, p) for p in df["filepath"]]
    caps = df["title"].astype(str).tolist()
    img_f, txt_f = [], []
    for s in range(0, len(paths), 256):
        x = torch.stack([preprocess(Image.open(p)) for p in paths[s:s + 256]]).to(device)
        img_f.append(F.normalize(model.encode_image(x).float(), dim=-1))
        txt_f.append(F.normalize(model.encode_text(tokenizer(caps[s:s + 256]).to(device)).float(), dim=-1))
    img_f, txt_f = torch.cat(img_f), torch.cat(txt_f)
    sims = img_f @ txt_f.T
    gt = torch.arange(len(sims), device=device)
    out = {}
    for name, s in (("i2t", sims), ("t2i", sims.T)):
        rank = (s > s[gt, gt][:, None]).sum(1)
        for k in (1, 5, 10):
            out[f"val_{name}_R@{k}"] = (rank < k).float().mean().item()
    out["val_mean_R@1"] = (out["val_i2t_R@1"] + out["val_t2i_R@1"]) / 2
    out["val_gap_norm"] = (img_f.mean(0) - txt_f.mean(0)).norm().item()
    out["val_matched_cos"] = sims.diagonal().mean().item()
    out["logit_scale"] = model.logit_scale.exp().item()
    model.train()
    return out


def log(record):
    print(json.dumps(record), flush=True)
    with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(ARCH, pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer(ARCH)
if args.grad_checkpointing:
    model.set_grad_checkpointing(True)

train_set = NegClipTsv(args.train_tsv, preprocess_train, tokenizer, args.hard_images, args.hard_captions)
g = torch.Generator()
g.manual_seed(args.seed)
loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=cuda, persistent_workers=args.workers > 0, generator=g)

exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n  # noqa: E731
params = list(model.named_parameters())
optimizer = torch.optim.AdamW(
    [{"params": [p for n, p in params if exclude(n, p)], "weight_decay": 0.0},
     {"params": [p for n, p in params if not exclude(n, p)], "weight_decay": args.wd}],
    lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=cuda)
total_steps = args.epochs * len(loader)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    e, es = step - args.warmup, total_steps - args.warmup
    return 0.5 * (1 + math.cos(math.pi * e / es)) * args.lr


with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(dict(vars(args), out_dir=out_dir, steps_per_epoch=len(loader), arch=ARCH,
                   torch=torch.__version__, open_clip=open_clip.__version__,
                   gpu=torch.cuda.get_device_name(0) if cuda else "cpu"), f, indent=2)

log(dict(epoch=0, step=0, **validate(model, preprocess_val, tokenizer)))
step = 0
model.train()
for epoch in range(1, args.epochs + 1):
    t0, running = time.time(), []
    for image, text, neg, h_image, h_text, h_neg in loader:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        images = [image] + ([h_image] if args.hard_images else [])
        texts = [text] + ([h_text] if args.hard_images else [])
        if args.hard_captions:
            texts += [neg] + ([h_neg] if args.hard_images else [])
        images = torch.cat(images).to(device, non_blocking=True)
        texts = torch.cat(texts).to(device, non_blocking=True)
        if step == 0:
            print(f"batch layout: {len(images)} images x {len(texts)} texts", flush=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            img_f, txt_f, scale = model(images, texts)
            loss = loss_fn(img_f, txt_f, scale)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))

        running.append(loss.item())
        step += 1
        if step % 50 == 0:
            print(f"epoch {epoch} step {step}/{total_steps} loss {np.mean(running[-50:]):.4f} "
                  f"lr {lr_at(step):.2e} {(time.time() - t0) / len(running):.2f}s/step", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    log(dict(epoch=epoch, step=step, train_loss=float(np.mean(running)), epoch_time_s=time.time() - t0,
             **validate(model, preprocess_val, tokenizer)))
    if args.save_every_epoch:
        torch.save(model.state_dict(), os.path.join(out_dir, f"epoch{epoch}.pt"))
    if args.max_steps and step >= args.max_steps:
        break

torch.save(model.state_dict(), os.path.join(out_dir, "final.pt"))
print(f"saved {os.path.join(out_dir, 'final.pt')}")
