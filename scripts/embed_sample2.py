"""Embed the local positive image/caption pairs using the existing pipeline.

Usage (from the repository root, in the project's Python environment):
    python scripts/embed_sample2.py --models clip
    python scripts/embed_sample2.py --models clip negclip negclip_gelu

Writes results/embeddings/<model>/sample2_positive.npz and a companion
sample2_positive.rows.json. Row i in img, txt, and rows is the same pair.
The NPZ format, normalized embeddings, and cache behavior are unchanged.
NegCLIP variants require checkpoints/negclip_state_dict.pt.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["clip", "negclip"],
                        help="Model registry names or names of checkpoints/<name>.pt")
    parser.add_argument("--device", default=None, help="Defaults to CUDA if available, otherwise CPU")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0,
                        help="Image loader workers (default: 0, compatible with Windows)")
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 0:
        parser.error("--threads must be positive and --workers must be non-negative")

    import torch

    from src.data_sample2 import sample2
    from src.embed import cache_path, fingerprint, get_embeddings
    from src.models import default_device, load_model

    torch.set_num_threads(args.threads)
    # Match scripts/embed_all.py's full-fp32 settings.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = args.device or default_device()
    bench = sample2()
    for model_name in args.models:
        print(f"[{model_name} / {bench['name']}] {len(bench['images'])} positive pairs", flush=True)
        start = time.time()
        model, preprocess, tokenizer = load_model(model_name, device=device)
        img, txt, scale = get_embeddings(model_name, bench, model, preprocess, tokenizer,
                                         workers=args.workers)
        path = cache_path(model_name, bench["name"])
        rows_path = os.path.splitext(path)[0] + ".rows.json"
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(dict(model=model_name, benchmark=bench["name"],
                           fingerprint=fingerprint(bench), rows=bench["rows"]),
                      f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[{model_name} / {bench['name']}] done in {time.time() - start:.0f}s -> {path} "
              f"(img {img.shape}, txt {txt.shape}, logit scale {scale:.1f})", flush=True)
        print(f"Row mapping -> {rows_path}", flush=True)
        del model, preprocess, tokenizer, img, txt


if __name__ == "__main__":
    main()
