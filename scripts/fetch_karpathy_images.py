"""Fetch only the Karpathy val + test images (10k) out of the remote val2014.zip.

Usage: .venv/bin/python scripts/fetch_karpathy_images.py data/coco [--workers 16]

The full val2014.zip is 6.2 GB / 40k images. We need only the 10k images in
the Karpathy val and test splits, so instead of downloading the whole zip we
read its central directory over HTTP range requests and pull each member
individually (~1.6 GB total). Already-present images are skipped, so the
script can be re-run after an interruption.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from remotezip import RemoteZip
from tqdm import tqdm

URL = "http://images.cocodataset.org/zips/val2014.zip"

parser = argparse.ArgumentParser()
parser.add_argument("coco_dir")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

needed = set()
for split in ("val", "test"):
    with open(os.path.join(args.coco_dir, f"coco_karpathy_{split}.json")) as f:
        needed.update(item["image"] for item in json.load(f))  # e.g. "val2014/COCO_val2014_....jpg"

todo = sorted(m for m in needed if not os.path.exists(os.path.join(args.coco_dir, m)))
print(f"{len(needed)} images needed, {len(needed) - len(todo)} already present, {len(todo)} to fetch")

local = threading.local()


def fetch(member):
    # one RemoteZip (and HTTP session) per thread
    if not hasattr(local, "zf"):
        local.zf = RemoteZip(URL)
    out = os.path.join(args.coco_dir, member)
    data = local.zf.read(member)
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out)


os.makedirs(os.path.join(args.coco_dir, "val2014"), exist_ok=True)
failed = []
with ThreadPoolExecutor(args.workers) as pool:
    futures = {pool.submit(fetch, m): m for m in todo}
    for fut in tqdm(as_completed(futures), total=len(futures), mininterval=10):
        if fut.exception() is not None:
            failed.append(futures[fut])

if failed:
    raise SystemExit(f"{len(failed)} images failed (e.g. {failed[0]}); re-run to retry")
print(f"All {len(needed)} Karpathy val + test images present in {os.path.join(args.coco_dir, 'val2014')}")
