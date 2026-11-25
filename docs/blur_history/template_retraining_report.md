# MIDV template retraining: Wiener and Restormer

Both paths were trained independently on the new 160-identity template training split. Twenty identities were reserved for validation and twenty for testing. The 60 synthetic test images contain three fixed blur/noise draws per test identity. All methods use identical test inputs, supplied OCR field geometry and Tesseract eng/PSM7.

**Handmade CER is provisional:** 113 photos / 19 annotated sides, with F12 excluded. The supplied field transcriptions await user audit. No handmade training or checkpoint selection occurred.

## Synthetic test: same 60 inputs for every method

| Method | CER | Mild CER | Medium CER | Strong CER | SSIM | Field SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Original | 31.74% | 10.01% | 22.98% | 62.23% | 0.6456 | 0.5662 |
| Previous Wiener shared | 37.40% | 23.24% | 28.92% | 60.04% | 0.5520 | 0.4378 |
| Previous Wiener selected | 36.32% | 19.56% | 29.37% | 60.04% | 0.5816 | 0.4798 |
| New Wiener shared | 30.04% | 14.72% | 16.20% | 59.20% | 0.5788 | 0.4923 |
| New Wiener selected | 28.73% | 12.91% | 22.72% | 50.55% | 0.6071 | 0.5559 |
| Restormer pretrained | 30.47% | 9.88% | 21.56% | 59.97% | 0.6651 | 0.5724 |
| Restormer previous 200-step pilot | 24.73% | 9.62% | 22.53% | 42.03% | 0.6947 | 0.6138 |
| Restormer new 2,000-step run | 11.86% | 12.72% | 10.91% | 11.94% | 0.8449 | 0.8941 |
| Sharp reference OCR control | 13.30% | 13.30% | 13.30% | 13.30% | 1.0000 | 1.0000 |

CER is pooled character edits/reference length; lower is better. SSIM is the image mean; higher is better. All synthetic outputs are quantized to 8-bit before scoring. Mild/medium/strong describe added disk-radius bands, not inferred real blur severity. Sharp-reference OCR is a control, not an assumption of zero errors: restoring background texture and crisp edges does not always improve this OCR configuration.

## Handmade benchmark: transfer to real captures

| Method | All blurry CER | Blurry/diffuse | Blurry/glare | Sharp/diffuse | Sharp/glare |
|---|---:|---:|---:|---:|---:|
| Original | 75.53% | 65.51% | 80.67% | 5.47% | 30.50% |
| Previous Wiener shared | 74.35% | 68.55% | 77.33% | 6.06% | 26.55% |
| Previous Wiener selected | 75.49% | 68.55% | 79.05% | 3.61% | 30.43% |
| New Wiener shared | 67.99% | 53.44% | 75.46% | 3.05% | 31.41% |
| New Wiener selected | 68.42% | 54.26% | 75.68% | 6.62% | 32.64% |
| Restormer pretrained | 79.36% | 72.46% | 82.90% | 6.06% | 31.41% |
| Restormer previous 200-step pilot | 78.91% | 72.30% | 82.30% | 6.54% | 33.31% |
| Restormer new 2,000-step run | 80.71% | 75.04% | 83.62% | 9.91% | 35.06% |

Handmade SSIM is unavailable: the sharp-camera references have not been verified for pixel-accurate alignment. Saved previous-model results were reused only after matching input hashes, field geometry, reference text, source code, OCR runtime and model hashes. Original OCR was freshly recomputed and matched on every image.

## Wiener decision

The selected bank changes blurry-photo CER from **75.49% to 68.42%** (-7.07 percentage points).
The candidate does not worsen pooled blurry CER. Inspect each capture condition and sharp controls before adopting it.
The default configuration was never overwritten, so no rollback is necessary. Both candidates and their provenance remain available.

The new shared filter has **67.99%** blurry-photo CER, versus **68.42%** for the new selected bank. Sharp controls and glare conditions must also be considered; this comparison does not establish that every input should be filtered.

## Restormer transfer result

Synthetic test CER falls from **24.73% to 11.86%**, but blurry handmade CER rises from **78.91% to 80.71%**. Do not promote the new checkpoint for real-photo OCR on this evidence. The stronger synthetic restoration result has not transferred to the handmade benchmark. This is consistent with a training/capture mismatch, but this experiment does not isolate its cause. The previous checkpoint and pretrained default remain unchanged.

## Training and interpretation

- Restormer: official pretrained weights → 2,000 AdamW steps, learning rate 1e-5, L1 loss, 128px patches. Checkpoint selected by lowest validation patch L1; final test and handmade CER were not used for selection.
- Wiener: 480 fixed training pairs (160 identities × three blur bands), unchanged 48-setting radius/K grid, no denoising or sharpening. Maximize mean field SSIM among candidates within 0.5 percentage points of minimum training CER. Train-only sharpness tertiles define experts; validation CER selects routing thresholds. Known synthetic radius is never used by the inference router.
- Wiener selection labels and reported synthetic severity are different: the former are image sharpness proxies; the latter are known added radii. No mild/medium/strong labels are assigned to handmade photos.
- This changes data domain and Restormer training budget as well as data count. It is not a controlled data-volume-only ablation. New digital templates differ from the previous relatively sharp camera-frame targets.
- There are five shared layouts across splits. More text identities do not establish unseen-layout generalization. The handmade set is a reused external benchmark, not an untouched final test.
- No images or old defaults were replaced. Parameter searches do not consult the synthetic test or handmade labels.

## Artifacts

- [New Wiener configuration](../wiener_midv_templates_training/frozen_config.json)
- [Restormer training run](../restormer_midv_templates_2000/run.json)
- [New Restormer checkpoint](../restormer_midv_templates_2000/best.pt)
- [Full metrics](summary.json), [comparability checks](verification.json), [Wiener decision](wiener_decision.json)
- Per-field predictions and review strips are in the four family/dataset subfolders.

## Wiener parameter changes

| Filter | Previous radius | Previous K | New radius | New K |
|---|---:|---:|---:|---:|
| shared | 4 | 0.3 | 1 | 1.0 |
| mild | 2 | 1.0 | 1 | 0.1 |
| medium | 4 | 0.3 | 1 | 1.0 |
| strong | 4 | 0.3 | 6 | 0.01 |

Routing thresholds also changed; see the frozen configurations. Expert names are sharpness groups, not externally verified blur classes. [Qualitative inspection](qualitative_review.json) records example-specific observations.
