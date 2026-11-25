"""Fetch only the COCO val2017 images SugarCrepe's 7 subsets reference.

Usage: .venv/bin/python scripts/fetch_sugarcrepe_images.py data/sugarcrepe [--workers 16]

val2017.zip is 778 MB / 5,000 images; SugarCrepe's 7,511 test cases reference
a subset of them (many images repeat across subsets). We read the zip's
central directory over HTTP range requests and pull only the needed members,
same approach as scripts/fetch_karpathy_images.py for val2014.
"""
import argparse
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from remotezip import RemoteZip
from tqdm import tqdm

URL = "http://images.cocodataset.org/zips/val2017.zip"

parser = argparse.ArgumentParser()
parser.add_argument("sugarcrepe_dir")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

needed = set()
for path in sorted(glob.glob(os.path.join(args.sugarcrepe_dir, "*.json"))):
    with open(path) as f:
        ann = json.load(f)
    needed.update(f"val2017/{item['filename']}" for item in ann.values())

out_dir = os.path.join(args.sugarcrepe_dir, "val2017")
os.makedirs(out_dir, exist_ok=True)
todo = sorted(m for m in needed if not os.path.exists(os.path.join(args.sugarcrepe_dir, m)))
print(f"{len(needed)} images needed, {len(needed) - len(todo)} already present, {len(todo)} to fetch")

local = threading.local()


def fetch(member):
    if not hasattr(local, "zf"):
        local.zf = RemoteZip(URL)
    out = os.path.join(args.sugarcrepe_dir, member)
    data = local.zf.read(member)
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out)


failed = []
with ThreadPoolExecutor(args.workers) as pool:
    futures = {pool.submit(fetch, m): m for m in todo}
    for fut in tqdm(as_completed(futures), total=len(futures), mininterval=10):
        if fut.exception() is not None:
            failed.append(futures[fut])

if failed:
    raise SystemExit(f"{len(failed)} images failed (e.g. {failed[0]}); re-run to retry")
print(f"All {len(needed)} SugarCrepe images present in {out_dir}")
