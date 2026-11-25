# Does closing the modality gap help hard-negative-trained CLIP? Final results

Setup: CLIP ViT-B/32 fine-tuned on COCO for 5 epochs in five configurations: plain, plain with
batch 512 (batch-size control), + hard captions, + hard images, and both (the NegCLIP recipe).
Each configuration was trained at learning rates 1e-5, 5e-6 and 1e-6 with seed 0, and at 5e-6
and 1e-6 with seeds 1 and 2 as well (30 runs in total).
Reference models are pretrained CLIP and the released NegCLIP. The gap vector Δ is fitted on
COCO val (Karpathy split), and every shift is evaluated on COCO test retrieval, COCO-Order,
VG-Relation, VG-Attribution and SugarCrepe. Numbers are clean accuracy (no noise) unless stated.
Tables and figures referred to below are in this folder.

## Answer

**No. Closing the gap post hoc hurts every model trained with hard captions.** Shrinking the
remaining gap on COCO val from about 0.85 to 0.10 with the text shift (`table_gap_closing_delta`,
`fig_shift_text_mean`) changes accuracy as follows:

| model | COCO-Order | VG-Relation | VG-Attribution | COCO i2t R@1 |
|---|---|---|---|---|
| released NegCLIP | −0.284 | −0.015 | −0.049 | −0.188 |
| FT NegCLIP recipe | −0.338 | −0.047 | −0.058 | −0.192 |
| FT + hard captions | −0.211 | −0.023 | −0.036 | −0.135 |
| FT plain (no hard negatives) | +0.008 | **+0.114** | +0.004 | −0.161 |
| CLIP | −0.216 | +0.017 | −0.023 | −0.165 |

The only gains are on VG-Relation, and only for models without hard captions (FT plain +0.11,
batch-512 +0.10, hard images +0.05). Those models start near or below CLIP on that benchmark.
Retrieval drops by 0.13–0.20 R@1 for every model.

## Why: the shift works only by suppressing one component of each caption

1. **Without re-normalisation the shift changes nothing.** The `text_norenorm` shift
   (y + αΔ, not projected back to the sphere) leaves every accuracy unchanged to 4 decimals,
   for all 8 models on all 4 case benchmarks (`table_h3_controls`). The shift adds the same
   amount to every caption's score for an image, so any effect has to come from the
   re-normalisation.
2. **That effect equals removing each caption's deviation along the gap direction.** The `orth`
   control first subtracts each caption's own deviation along û = Δ/|Δ|. Doing only that,
   with no shift at all (α = 0), reproduces the full gap-closing change almost exactly
   (`table_gap_component_removed` vs `table_h3_controls`, text at α = 1):

   | COCO-Order | remove deviation along û only | close the gap (text shift, α = 1) |
   |---|---|---|
   | released NegCLIP | −0.327 | −0.327 |
   | FT NegCLIP recipe | −0.364 | −0.385 |
   | CLIP | −0.223 | −0.231 |
   | FT plain, VG-Relation | +0.143 | +0.125 |

   Re-normalising after adding a large shared offset shrinks each caption's position along û
   relative to everything else. Closing the gap therefore amounts to deleting a single
   direction from the text embeddings.
3. **For hard-caption models that direction carries compositional information.** It holds only
   about 0.7 % of text variance (`var_frac_txt` in `table_gap_metrics`), yet removing it costs
   NegCLIP-style models 0.21–0.36 on COCO-Order. In models without hard captions the same
   direction works against VG-Relation, which is why only they gain there.

## What hard negatives do to the gap during training

- **Hard images shrink the gap. Batch size does not.** At lr 5e-6 the final COCO-val gap is
  0.650 with hard images and 0.640 with the NegCLIP recipe. It is 0.775 for plain fine-tuning
  and 0.799 for the batch-512 control (`table_val_final_epoch`, `fig_training_curves`). The
  control matches plain fine-tuning on every benchmark (within 0.015, `table_batch_control`),
  so the larger image batch does not explain the hard-image effect.
- **Gap size and compositional skill are unrelated.** Hard captions barely move the gap (0.758)
  but give the best compositional scores. Hard images give the smallest gap but
  plain-level compositional scores.

## Matched learning rates and seeds

- The ranking is the same at every learning rate (`table_matched_lr`, `fig_matched_lr`), so the
  earlier learning-rate confound does not change any conclusion.
- **Hard captions alone are best on every compositional benchmark at every learning rate:**
  COCO-Order 0.87–0.89, VG-Relation 0.85–0.86, VG-Attribution 0.73–0.745. Released NegCLIP
  scores 0.825, 0.838 and 0.721.
- **Adding hard images to hard captions costs accuracy.** COCO-Order falls from 0.890 to 0.838
  at lr 1e-6 and from 0.878 to 0.781 at lr 5e-6, and retrieval drops at higher learning rates.
- The NegCLIP recipe at lr 1e-6 reproduces released NegCLIP: COCO-Order 0.838 vs 0.825,
  VG-Relation 0.840 vs 0.838, VG-Attribution 0.725 vs 0.721.
- **Seed noise is small.** The standard deviation over 3 seeds is at most 0.007 on every
  metric (`table_seed_variance`, `fig_seed_variance`). Differences between configurations are
  10–60 times larger.

## SugarCrepe (`table_sugarcrepe_clean`, `table_sugarcrepe_delta`, `fig_sugarcrepe`)

- **Clean:** hard captions are what lift the swap subsets (swap_att 0.70 → 0.78,
  swap_obj 0.67 → 0.78). Hard images add nothing there. Mean accuracy over subsets:
  FT + hard captions 0.837 > released NegCLIP 0.829 ≈ FT NegCLIP recipe 0.828 > FT plain 0.803
  > CLIP 0.739.
- **After closing the gap (α = 1):** accuracy rises on add_att (+0.09 to +0.22) and add_obj
  (+0.04 to +0.06), and falls on replace_att, replace_rel and the swap subsets (−0.03 to −0.11).
  These cancel to about zero in the pooled score. One untested explanation is that the
  direction along û correlates with caption length, since the "add" negatives are longer
  captions.

## Under embedding noise (`table_noise_delta`)

The harm from closing the gap shrinks as noise grows. For NegCLIP on COCO-Order it goes from
−0.28 with no noise to −0.05 at σ = 1. Closing the gap gives a small SugarCrepe gain only under
noise: +0.03 to +0.05 for every model at σ = 0.5. The noise robustness hypothesis therefore
holds only weakly, and only in the noisy regime.

## Caption naturalness (`table_naturalness`, `fig_naturalness`)

The correlation (Spearman ρ) between a COCO-Order caption's projection on û and its GPT-2
perplexity is small and its sign varies by model, from −0.18 (FT + hard captions) to +0.06
(FT plain). The gap direction does not encode how natural a caption reads, so the earlier
naturalness explanation is not supported.

## Caveats

- One architecture (ViT-B/32) and 5 epochs of fine-tuning.
- The batch-512 control has one seed per learning rate.
- "Best" learning rate per configuration is picked by COCO-val R@1. That puts hard images and
  the NegCLIP recipe at 1e-6, where the gap barely moves, so `table_gap_metrics` (selected
  checkpoints) shows gaps of about 0.88 for them. The matched-LR tables cover every rate.
