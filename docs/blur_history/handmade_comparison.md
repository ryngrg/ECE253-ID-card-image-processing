# Handmade ID-card OCR comparison

**Provisional: assistant-transcribed field references await user audit.**

113 of 115 photos, 19 annotated card sides, 129 unique printed fields, and 785 field observations per method. Two F12 photos are excluded pending user transcription. Front/back physical-card identities are not inferred.

Each filter starts from the same unprocessed native-resolution ROI. The two filter paths are evaluated independently, not chained. Tesseract eng/PSM7 and field geometry are identical across methods. Original OCR predictions were verified identical between the two independent runs. No handmade photos were used for fitting or model selection.

| Capture condition | Photos | Original CER | Wiener shared | Wiener selected | Restormer pretrained | Restormer synthetic-tuned |
|---|---:|---:|---:|---:|---:|---:|
| Blurry / diffuse | 19 | 65.51% | 68.55% | 68.55% | 72.46% | 72.30% |
| Blurry / glare | 37 | 80.67% | 77.33% | 79.05% | 82.90% | 82.30% |
| Sharp / diffuse control | 19 | 5.47% | 6.06% | 3.61% | 6.06% | 6.54% |
| Sharp / glare control | 38 | 30.50% | 26.55% | 30.43% | 31.41% | 33.31% |

Lower CER is better. A percentage-point change is after minus before. CER is pooled character edit distance divided by pooled reference length, not an average of image percentages. Case and punctuation are retained after Unicode NFKC and whitespace normalization; blank OCR is counted as deletions.

## Separate path reports

- Wiener: [reviewed_metrics/REPORT.md](wiener/reviewed_metrics/REPORT.md)
- Restormer: [reviewed_metrics/REPORT.md](restormer/reviewed_metrics/REPORT.md)

The path reports include WER, exact-field accuracy and runtime. Per-image JSON files include every field reference, prediction and edit count. Review strips are in each original inference folder, grouped by method.

## Interpretation and proposal coverage

- The current Wiener implementation is the frozen MIDV-trained shared filter and sharpness-selected expert bank. It is not the original proposal’s noise-adaptive per-image radius-sweep ablation.
- Restormer uses the official single-image defocus weights and the existing 200-step MIDV synthetic fine-tuning pilot, with identical 512px tiles and 64px overlap. No real-handmade fine-tuning has been performed.
- This measures the restoration-plus-supplied-field OCR pipeline. It is not automatic card/field detection or whole-card transcription.
- The two B/S labels and D/G conditions come from filenames. Mild/medium/strong and clipped-core/halo labels have not been established.
- PSNR, SSIM and LPIPS remain unreported: 19 diffuse pair registration candidates and review montages were prepared, but pixel-accurate alignment has not been certified. Geometry good enough to locate a field is not necessarily good enough for a pixel metric.
- Repeated captures of the same sides and uncertain front/back mappings limit generalization claims. Printed mock-template overlap with public training data has not been exhaustively audited. These are fixed-model external-photo results, not proof of unseen-identity performance.
- User review of the reference table is required before presenting these CER values as final. Text-only corrections can be rescored from saved predictions; geometry or scope changes require a new evaluation.

## Per-image changes

| Method | Blurry images improved | Unchanged | Worsened |
|---|---:|---:|---:|
| Wiener shared | 34 | 1 | 21 |
| Wiener selected | 35 | 1 | 20 |
| Restormer pretrained | 20 | 8 | 28 |
| Restormer synthetic-tuned | 26 | 6 | 24 |

## Blurry diffuse results by annotated card side

| Side | Photos | Original CER | Wiener shared | Wiener selected | Restormer pretrained | Restormer synthetic-tuned |
|---|---:|---:|---:|---:|---:|---:|
| B01 | 1 | 40.74% | 33.33% | 33.33% | 40.74% | 40.74% |
| B02 | 1 | 90.00% | 80.91% | 80.91% | 94.55% | 95.45% |
| B03 | 1 | 74.53% | 38.68% | 38.68% | 74.53% | 66.04% |
| B04 | 1 | 56.04% | 57.88% | 57.88% | 78.75% | 75.46% |
| B05 | 1 | 52.17% | 76.47% | 76.47% | 67.18% | 77.09% |
| B06 | 1 | 87.73% | 88.96% | 88.96% | 82.21% | 82.82% |
| B07 | 1 | 27.08% | 24.31% | 24.31% | 40.28% | 19.44% |
| B08 | 1 | 74.81% | 76.30% | 76.30% | 82.96% | 85.19% |
| F01 | 1 | 57.14% | 51.02% | 51.02% | 59.18% | 46.94% |
| F02 | 1 | 81.82% | 82.83% | 82.83% | 78.79% | 70.71% |
| F03 | 1 | 93.02% | 83.72% | 83.72% | 93.02% | 88.37% |
| F04 | 1 | 81.00% | 77.00% | 77.00% | 81.00% | 79.00% |
| F05 | 1 | 91.95% | 74.71% | 74.71% | 88.51% | 83.91% |
| F06 | 1 | 94.57% | 78.26% | 78.26% | 89.13% | 92.39% |
| F07 | 1 | 100.00% | 93.68% | 93.68% | 97.89% | 97.89% |
| F08 | 1 | 0.00% | 22.73% | 22.73% | 2.27% | 2.27% |
| F09 | 1 | 26.32% | 47.37% | 47.37% | 15.79% | 28.95% |
| F10 | 2 | 70.53% | 72.11% | 72.11% | 74.21% | 72.63% |
