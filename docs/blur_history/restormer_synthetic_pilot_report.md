# Restormer synthetic pilot comparison

200 optimizer steps, learning rate 1e-5, L1 loss; official single-image defocus weights as initialization. Training: 20 MIDV cards. Validation: 5 cards. Test: 5 cards × 3 fixed blur/noise conditions = 15 images. All frames from a card stay in one split.

Native-resolution tiled evaluation on MPS, tile 512px / overlap 64px, identical for both models. CER uses the existing Tesseract field evaluator (PSM 7); 1,158 reference characters across repeated blur conditions.

| Method | CER ↓ | SSIM ↑ | Field SSIM ↑ | RGB L1 ↓ |
|---|---:|---:|---:|---:|
| Unfiltered | 63.73% | 0.8874 | 0.8504 | 0.01497 |
| Pretrained Restormer | 53.71% | 0.9119 | 0.8724 | 0.02115 |
| Fine-tuned Restormer | 49.48% | 0.9405 | 0.9099 | 0.01112 |
| Reference-frame control | 44.82% | 1.0000 | 1.0000 | 0.00000 |

| Added disk-blur band | Unfiltered CER | Pretrained CER | Fine-tuned CER | Unfiltered SSIM | Pretrained SSIM | Fine-tuned SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Mild | 47.15% | 46.63% | 44.56% | 0.9547 | 0.9518 | 0.9737 |
| Medium | 51.55% | 48.45% | 43.52% | 0.8922 | 0.9143 | 0.9399 |
| Strong | 92.49% | 66.06% | 60.36% | 0.8153 | 0.8696 | 0.9079 |

Fine-tuning improves pooled CER by 4.23 percentage points relative to the pretrained model (7.9% relative error reduction). The larger unfiltered-to-fine-tuned reduction is 14.25 percentage points. Validation patch L1 fell from 0.036013 to 0.025652; step 200 was selected without using test scores.

## Limits

- This is synthetic-only fine-tuning. No registered real blur/sharp pair was used for training. The 18 enabled real capture pairs still need alignment, identity confirmation, split assignment, and valid ROIs.
- Targets are the sharpest available camera frame per MIDV card, not certified blur-free originals. Their 44.82% CER shows that OCR and reference quality still limit results. SSIM measures recovery of additional synthetic degradation.
- Five test identities are too few for a broad generalization claim, and these identities appeared in earlier project experiments.
- Some individual images regress in CER despite better SSIM. Pooled improvement does not mean every image improves.
- Mild/medium/strong denotes sampled added disk radius (0.5–2.5 / 2.5–5 / 5–8 native pixels), not the earlier real sharpness-based groups. These numbers cannot be directly compared with the older Wiener tables.
- Field geometry is supplied by annotations; field detection is not evaluated.

Files: `summary.json`, `per_image.json`, `run.json`; processed images are grouped by method. The qualitative crop sheet is `review/strong_text_fields.png`.
