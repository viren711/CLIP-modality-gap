#!/usr/bin/env bash
# Build the self-contained package for the GPU server: code + all local data
# + the NegCLIP weights. COCO train2014 / full val2014 are NOT included (19 GB);
# run_gpu.sh downloads them on the server.
#   bash scripts/make_gpu_package.sh   ->   dist/cs725_gpu_package.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dist/cs725_gpu_package.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# stage via symlinks so nothing is copied twice on a nearly-full disk
PKG="$STAGE/cs725_project"
mkdir -p "$PKG/checkpoints"
for f in run_gpu.sh requirements.txt README.md plan.txt approaches.txt src scripts data; do
    ln -s "$ROOT/$f" "$PKG/$f"
done
ln -s "$ROOT/checkpoints/negclip_state_dict.pt" "$PKG/checkpoints/negclip_state_dict.pt"

mkdir -p "$ROOT/dist"
rm -f "$OUT"
# -1: fast (jpgs do not compress anyway); follow symlinks; skip caches and partial files
(cd "$STAGE" && zip -r -q -1 "$OUT" cs725_project -x '*/__pycache__/*' '*.part' '*/.val2014_done')
ls -lh "$OUT"
echo "Copy with:  scp $OUT user@server:~/   then:  unzip cs725_gpu_package.zip && cd cs725_project && bash run_gpu.sh"
