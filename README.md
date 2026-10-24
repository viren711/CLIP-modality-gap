# Does closing the modality gap help hard-negative-trained CLIP?

CS 725 course project · Viren Mehta (25M2132), Lavanay Thakral (25D0381), Shrey Patel (26M1104), Yash Chopra (26M1520)

CLIP keeps image and text embeddings in two separate regions of its shared space,
known as the **modality gap** (Liang et al., 2022). **NegCLIP** (Yuksekgonul et al., 2023)
fine-tunes CLIP with hard negative captions and images so that it understands word
order and relations. This project measures the gap in CLIP, NegCLIP and four
controlled fine-tunes. It then shrinks or widens the gap after training and tests
what that does to compositionality, retrieval and robustness to noise.

| | |
|---|---|
| Report (15 pages) | [report.pdf](report.pdf) |
| Mid-project slides, PDF and talk transcript | `midterm_presentation/` (kept locally, not in git) |
| Original proposal | [CLIP_Modality_Gap_Proposal.pdf](CLIP_Modality_Gap_Proposal.pdf) |
| All planned approaches | [approaches.txt](approaches.txt) |
| Detailed work log: every decision, check and number | [plan.txt](plan.txt) |

## Findings so far

1. **Hard-negative training does not shrink the gap.** NegCLIP's gap is 0.855 vs 0.832
   for CLIP. Hard captions reshape the text side instead: the text cone widens and
   random image–caption pairs drop from 0.15 to 0.08 cosine similarity.
2. **Compositionality and the gap are decoupled.** Compositionality comes from hard
   negative *captions*. The size of the gap moves with hard negative *images* and the
   learning rate: −0.12 vs −0.02 after 5 epochs at LR 5e-6.
3. **Closing the gap is not free.** On clean data, closing half of it costs NegCLIP-style
   models 0.07–0.14 COCO-Order accuracy. The cause is re-normalisation: it favours
   word-shuffled captions, which lie further along the gap direction.
4. **Under noise, a partly closed gap wins.** With Gaussian noise on the embeddings,
   every model retrieves best with the gap at about 0.6–0.7 (α ≈ 0.25–0.5). The gain
   is slightly larger for hard-negative models.
5. **Side finding.** NegCLIP was trained with QuickGELU but evaluated with GELU in its
   paper's code. Only that mismatch reproduces the published COCO-Order score (0.868 vs
   0.825 as trained).

Clean results (COCO Karpathy test and ARO, no shift):

| Model | COCO-Order | VG-Attribution | VG-Relation | COCO i2t R@1 | COCO t2i R@1 | Gap ‖Δ‖ |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 0.472 | 0.632 | 0.638 | 0.502 | 0.302 | 0.832 |
| `ft_plain` (no hard negatives) | 0.291 | 0.647 | 0.542 | 0.618 | 0.454 | 0.780 |
| `ft_hardimg` (hard images) | 0.303 | 0.650 | 0.554 | 0.607 | 0.451 | 0.888 |
| `ft_hardcap` (hard captions) | 0.878 | 0.738 | 0.860 | 0.599 | 0.454 | 0.765 |
| `ft_negclip` (our reproduction) | 0.838 | 0.725 | 0.840 | 0.597 | 0.449 | 0.883 |
| NegCLIP (released) | 0.825 | 0.721 | 0.838 | 0.593 | 0.448 | 0.855 |

**Known caveat:** `ft_plain` and `ft_hardcap` were selected at LR 5e-6, the other two at 1e-6.
Matched-LR retraining is the first item for the final presentation (see the report, Section 6).

## How it works

```
data ──► encode once per model ──► cached embeddings ──┬─► 01 measure the gap
         (scripts/embed_all.py)    (results/embeddings) ├─► 02 shift the gap (α sweep, target gaps)
                                                        └─► 04 evaluate: ARO + COCO retrieval,
                                                               clean and under embedding noise
GPU:  scripts/train_finetune.py ──► 2×2 grid of controls ──► same pipeline
```

- **Gap:** Δ = mean(image embeddings) − mean(text embeddings), fitted on Karpathy val.
- **Shift:** y′ = (y + αΔ) / ‖y + αΔ‖. α = 0 is the original model, α = 1 closes the gap,
  α < 0 widens it. There is also a symmetric variant, a median Δ, and target-gap mode.
- **Controls:** one script re-implements the NegCLIP recipe, with two switches:

  | | no hard captions | hard negative captions |
  |---|---|---|
  | **no hard images** | `ft_plain` (CLIP-FT control) | `ft_hardcap` |
  | **hard negative images** | `ft_hardimg` | `ft_negclip` (NegCLIP reproduction) |

## Repository layout

```
src/
  common.py       paths, BLIP/ARO caption cleaning, helpers
  models.py       model registry: clip, negclip, negclip_gelu, any checkpoints/<name>.pt
  data.py         benchmarks (COCO Karpathy, VG-Relation, VG-Attribution, COCO-Order) in one index format
  embed.py        encode once per model + cache; exact 5x faster text encoder (padding trimmed)
  gap.py          gap metrics (Approach 01) and gap shifts (Approach 02)
  evaluate.py     ARO accuracy / margin, COCO Recall@K, embedding noise
scripts/
  download_data.sh            all evaluation data + NegCLIP checkpoint (resumable)
  fetch_karpathy_images.py    pulls only the 10k needed images out of the remote val2014.zip
  convert_negclip.py          NegCLIP release -> safe weights-only state_dict
  check_overlap.py            verifies no evaluation image was in NegCLIP's training data
  embed_all.py                encode + cache (CPU or GPU)
  measure_gap.py              -> results/gap_metrics.csv, PCA figures
  shift_eval.py               -> results/shift_eval.csv
  plot_shift.py               metric vs remaining gap figures
  run_midterm.sh              the CPU pipeline, in order
  train_finetune.py           NegCLIP-recipe fine-tuning with --hard-images / --hard-captions
  gpu/                        sweep launcher, LR selection, full COCO download
  make_gpu_package.sh         builds dist/cs725_gpu_package.zip for the GPU server
run_gpu.sh                    the one command to run on a GPU server
results/                      CPU results: clip + negclip (csv, figures)
gpu_results/                  GPU results: all 7 models, fine-tuning logs, figures
```

Git-ignored and re-creatable: `data/`, `checkpoints/`, `results/embeddings/`, `.venv/`,
`dist/`, `logs/`, `midterm_presentation/`, and `gpu_results/results_gpu.tar.gz` (3.6 GB, includes the GPU embeddings).

## Setup

```bash
python3 -m venv .venv                       # without python3-venv: --without-pip, then get-pip.py
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm          # only to rebuild the COCO-Order cache
.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"
```

On a GPU machine, install the torch build that matches its driver instead of the CPU wheel
(for example `--index-url https://download.pytorch.org/whl/cu126`).

## Reproduce

**Evaluation on CPU** (CLIP and NegCLIP, about 3 GB of data, about 5 h, mostly encoding):

```bash
bash scripts/download_data.sh
bash scripts/run_midterm.sh                 # overlap check -> encode -> gap -> shift sweep -> plots
```

**Fine-tuned controls on a GPU server** (12 runs took 2.2 h on 3 × 48 GB GPUs):

```bash
bash scripts/make_gpu_package.sh            # -> dist/cs725_gpu_package.zip (code + data + NegCLIP weights)
scp dist/cs725_gpu_package.zip user@server:/some/dir/
# on the server
unzip cs725_gpu_package.zip && cd cs725_project
GPUS="0 1 3" CUDA_DEVICE_ORDER=PCI_BUS_ID PRUNE_RUNS=1 \
  TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash run_gpu.sh
# bring back results_gpu.tar.gz
```

`run_gpu.sh` downloads the full COCO 2014 images (19 GB) and smoke-tests the heaviest
config. It then runs the learning-rate sweep, picks the best LR per config, and evaluates
all 7 models. Every step is resumable, and all caches stay inside the package folder.

## Data

| Data | Size | Used for |
|---|---|---|
| COCO Karpathy val | 5,000 images, 25,010 captions | fitting Δ (held out) |
| COCO Karpathy test | 5,000 images, 25,010 captions | gap measurement, retrieval |
| ARO VG-Relation / VG-Attribution | 23,937 / 28,748 cases | relation and attribute binding |
| ARO COCO-Order | 25,010 cases (true + 4 shuffled captions) | word-order sensitivity |
| NegCLIP train split | 109,468 COCO images with hard negatives | fine-tuning the controls |

None of the evaluation images appear in NegCLIP's training file (`scripts/check_overlap.py`).

## References

Radford et al. 2021 (CLIP) · Liang et al. 2022 (Mind the Gap) · Yuksekgonul et al. 2023
(ARO / NegCLIP, [code](https://github.com/mertyg/vision-language-models-are-bows)) ·
Chowers et al. 2026 (Bug or Feature?) · Kamath et al. 2024 (Hard Positive Truth) ·
Wang & Isola 2020 (alignment and uniformity) · [OpenCLIP](https://github.com/mlfoundations/open_clip)
