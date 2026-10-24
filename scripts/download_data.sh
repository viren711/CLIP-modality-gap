#!/usr/bin/env bash
# Downloads everything needed for the mid-project evaluation.
# Safe to re-run: every step is skipped if its output already exists,
# and partial downloads are written to *.part and only renamed when complete.
#
#   1. COCO 2014 caption annotations (train + val)
#   2. Karpathy splits (train / val / test json, BLIP format)
#   3. COCO val2014 images -> only the 10k Karpathy val + test images,
#      fetched straight out of the remote zip with HTTP range requests
#   4. ARO: Visual Genome Relation / Attribution annotations + images
#   5. NegCLIP checkpoint (ViT-B/32, Yuksekgonul et al. 2023)
#   6. NegCLIP train / valid split files (from the ARO repo)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COCO="$ROOT/data/coco"
ARO="$ROOT/data/aro"
CKPT="$ROOT/checkpoints"
PY="$ROOT/.venv/bin/python"
GDOWN="$ROOT/.venv/bin/gdown"

mkdir -p "$COCO/annotations" "$ARO" "$CKPT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# fetch URL OUT: resumable wget into OUT.part, renamed on success
fetch() {
    wget -c -nv -O "$2.part" "$1"
    mv "$2.part" "$2"
}

# gfetch DRIVE_ID OUT: Google Drive download into OUT.part, renamed on success
gfetch() {
    "$GDOWN" "https://drive.google.com/uc?id=$1" -O "$2.part"
    mv "$2.part" "$2"
}

# 1. COCO 2014 caption annotations --------------------------------------
if [[ ! -f "$COCO/annotations/captions_val2014.json" ]]; then
    log "Downloading COCO 2014 annotations"
    fetch http://images.cocodataset.org/annotations/annotations_trainval2014.zip "$COCO/annotations_trainval2014.zip"
    # only the caption files are needed (instances/keypoints are not)
    unzip -o -j "$COCO/annotations_trainval2014.zip" \
        annotations/captions_train2014.json annotations/captions_val2014.json \
        -d "$COCO/annotations"
    rm -f "$COCO/annotations_trainval2014.zip"
fi

# 2. Karpathy splits -----------------------------------------------------
for split in train val test; do
    f="$COCO/coco_karpathy_${split}.json"
    if [[ ! -f "$f" ]]; then
        log "Downloading Karpathy $split split"
        fetch "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_${split}.json" "$f"
    fi
done

# 3. COCO val2014 images (Karpathy val + test subset) -------------------
if [[ ! -f "$COCO/.val2014_done" ]]; then
    log "Fetching Karpathy val + test images from remote val2014.zip"
    "$PY" "$ROOT/scripts/fetch_karpathy_images.py" "$COCO" --workers 16
    touch "$COCO/.val2014_done"
fi

# 4. ARO: VG-Relation / VG-Attribution -----------------------------------
if [[ ! -f "$ARO/visual_genome_relation.json" ]]; then
    log "Downloading VG-Relation annotations"
    gfetch 1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3 "$ARO/visual_genome_relation.json"
fi
if [[ ! -f "$ARO/visual_genome_attribution.json" ]]; then
    log "Downloading VG-Attribution annotations"
    gfetch 13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw "$ARO/visual_genome_attribution.json"
fi
if [[ ! -d "$ARO/images" ]]; then
    log "Downloading VG-R / VG-A images (~1 GB)"
    gfetch 1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9 "$ARO/vgr_vga_images.zip"
    unzip -q "$ARO/vgr_vga_images.zip" -d "$ARO"
    rm -f "$ARO/vgr_vga_images.zip"
fi

# 5. NegCLIP checkpoint --------------------------------------------------
# The 1.8 GB release also contains optimizer state; keep only the weights.
if [[ ! -f "$CKPT/negclip_state_dict.pt" ]]; then
    if [[ ! -f "$CKPT/negclip.pth" ]]; then
        log "Downloading NegCLIP checkpoint"
        gfetch 1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ "$CKPT/negclip.pth"
    fi
    "$PY" "$ROOT/scripts/convert_negclip.py" "$CKPT/negclip.pth" "$CKPT/negclip_state_dict.pt"
    rm -f "$CKPT/negclip.pth"
fi

# 6. NegCLIP train / valid split files (overlap check, CLIP-FT training) --
NEG="$ROOT/data/negclip"
mkdir -p "$NEG"
for split in train valid; do
    f="$NEG/${split}_neg_clip.tsv"
    if [[ ! -f "$f" ]]; then
        log "Downloading NegCLIP $split split file"
        fetch "https://raw.githubusercontent.com/mertyg/vision-language-models-are-bows/main/temp_data/${split}_neg_clip.tsv" "$f"
    fi
done

log "All downloads finished"
