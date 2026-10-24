#!/usr/bin/env bash
# GPU server only: full COCO 2014 train + val images, needed to fine-tune on
# NegCLIP's training split (80,041 train2014 + 29,427 val2014 "restval" images).
# The package already holds the 10k Karpathy val/test images; unzip -n keeps them.
# Set KEEP_ZIPS=1 to keep the zips (13 GB + 6.2 GB) after extraction.
set -euo pipefail
COCO="$(cd "$(dirname "$0")/../.." && pwd)/data/coco"
mkdir -p "$COCO"

for split in train2014 val2014; do
    want=$([[ $split == train2014 ]] && echo 82783 || echo 40504)
    have=$(find "$COCO/$split" -name '*.jpg' 2>/dev/null | wc -l || true)  # dir may not exist yet
    if [[ "$have" -ge "$want" ]]; then
        echo "$split: $have images present"
        continue
    fi
    # an already-copied data/coco/<split>.zip is used as is (for servers without internet)
    if [[ ! -f "$COCO/$split.zip" ]]; then
        echo "$split: $have/$want images present, downloading"
        wget -c -nv -O "$COCO/$split.zip.part" "http://images.cocodataset.org/zips/$split.zip"
        mv "$COCO/$split.zip.part" "$COCO/$split.zip"
    fi
    unzip -q -n "$COCO/$split.zip" -d "$COCO"
    have=$(find "$COCO/$split" -name '*.jpg' | wc -l)
    [[ "$have" -ge "$want" ]] || { echo "ERROR: only $have/$want $split images after unzip"; exit 1; }
    [[ "${KEEP_ZIPS:-0}" == "1" ]] || rm -f "$COCO/$split.zip"
    echo "$split: $have images"
done
