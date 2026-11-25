"""Pick the learning rate per config by final-epoch COCO-val mean R@1 (as in the
NegCLIP paper) and export it as checkpoints/<config>.pt for the eval pipeline.

Usage: python scripts/gpu/select_best.py [--prune]
--prune deletes final.pt of the runs that were not selected (logs are kept).
Writes results/finetune_runs.csv (every run, every epoch) and
results/finetune_selection.json (chosen run per config).
"""
import glob
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = []
for log in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "log.jsonl"))):
    run = os.path.basename(os.path.dirname(log))
    cfg = json.load(open(os.path.join(os.path.dirname(log), "config.json")))
    for line in open(log):
        rows.append(dict(run=run, config=cfg["name"], lr=cfg["lr"], seed=cfg["seed"],
                         done=os.path.exists(os.path.join(os.path.dirname(log), "final.pt")), **json.loads(line)))
df = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
df.to_csv(os.path.join(ROOT, "results", "finetune_runs.csv"), index=False)

final = df[df.done].sort_values("epoch").groupby("run").tail(1)


def export(run, name):
    """checkpoints/<name>.pt as a hard link to the run's final.pt (no extra disk)."""
    src, dst = os.path.join(ROOT, "runs", run, "final.pt"), os.path.join(ROOT, "checkpoints", f"{name}.pt")
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return dst


# every finished run is a model of its own: checkpoints/<config>_lr<lr>_s<seed>.pt
for run in final.run:
    export(run, run)

selection = {}
for config, d in final.groupby("config"):
    # best LR judged on seed 0 only, so configs with extra seeds are not favoured
    d0 = d[d.seed == 0] if (d.seed == 0).any() else d
    best_lr = d0.groupby("lr")["val_mean_R@1"].mean().idxmax()
    best = d0[d0.lr == best_lr].iloc[0]
    dst = export(best.run, config)
    selection[config] = dict(run=best.run, lr=float(best_lr), val_mean_R1=float(best["val_mean_R@1"]),
                             val_gap_norm=float(best.val_gap_norm), checkpoint=dst)
    print(f"{config:11s} -> lr {best_lr:g}  val mean R@1 {best['val_mean_R@1']:.4f}  gap {best.val_gap_norm:.3f}")
json.dump(selection, open(os.path.join(ROOT, "results", "finetune_selection.json"), "w"), indent=2)

if "--prune" in sys.argv:
    keep = {v["run"] for v in selection.values()}
    for run in sorted(set(final.run) - keep):
        os.remove(os.path.join(ROOT, "runs", run, "final.pt"))
        print(f"pruned runs/{run}/final.pt")
