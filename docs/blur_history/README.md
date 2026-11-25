# Blur experiments done before the merge (archive, read-only)

Reports copied unchanged from the blur repo, `https://github.com/Mlazazarus/ECE253` @ `2893d66`, so that this repo is enough
to write the report. Paths inside them refer to that repo. Their metric is field CER (Tesseract, hand-placed
fields, native-resolution photos); ours is fixed-box character accuracy on rectified images. **Do not mix the numbers.**

## Why none of these models / parameters are used for our final numbers
Real-data pilot (`real_mixed_pilot_report.txt`; Restormer `step_002000.pt` and Wiener radius 3 / K 0.3):

| Used for | Pilot data | Overlap with our splits |
|---|---|---|
| training | B01, F01, F03, F05, F08 + 53 MIDV clip frames of 13 documents | **F03 = our test, F01 = our val**; `esp_id_90` is one of our 100 external MIDV clips |
| selection | B07, F09 + 16 MIDV frames of 3 documents | both cards are in our train split; no MIDV overlap |
| its test | B03, F04 (paired), + F10 | B03, F04 are our test cards (measured there, not tuned on) |

It also trained on MIDV, which our rules exclude, and at native resolution: the text area of B01 alone is 2335 px wide
in the photo, the whole rectified card 1012 px, so a radius of 3 px there is about 1 px in our images.
Several earlier comparisons were run on all handmade cards, test cards included, so the choice of method family is
not blind to the test cards. Our retuning is.

All 115 handmade photos of that repo are byte-identical to files in our `data/raw/` (we have 26 more: F11, F13, F14 and
most of F12).

## What was tried
| Experiment | Result | Report |
|---|---|---|
| Restormer fine-tuned on real pairs (pilot) | best result: CER on B_D 74% -> 30%, B_G 87% -> 58% (4 / 6 photos; B03, F04, F10) | `real_mixed_pilot_*` |
| Restormer fine-tuned on synthetic blur (200 and 2000 steps) | large gain on synthetic blur, worse than untreated on our blurry photos (81% vs 76%) | `restormer_synthetic_pilot_report.md`, `template_retraining_report.md` |
| Restormer pretrained | worse than untreated on our blurry photos (79% vs 76%) | `handmade_comparison.md`, `template_retraining_report.md` |
| Fixed disk Wiener | small gains (pilot: B_D 74% -> 67%) | `wiener_expanded_report.txt`, `real_mixed_pilot_*` |
| Wiener bank by sharpness (mild / medium / strong) | no useful gain over one shared filter | `wiener_expanded_report.txt` |
| Adaptive Wiener (radius by OCR confidence, K by noise; the proposal's method) | on MIDV the training picked "no filter"; small text-CER setup on HDC2021 | only logs; code ported |
| Directional (motion) Wiener | worse than the disk | `wiener_directional_report.txt` |
| Higher-order Wiener, larger disks | no real gain | `wiener_higher_order_report.txt` |
| Edge / unsharp / tone step after Wiener | worse than Wiener alone | `wiener_edge_sharpening_report.txt` |
| SSIM-constrained Wiener selection | same setting selected | `wiener_cer_ssim_report.txt` |
| Gaussian denoising before Wiener | not better on HDC2021 training (CER 57.5% without, 58.5% with) | only logs |

## What was ported, and what was not
| Blur repo | Here |
|---|---|
| `restormer/scripts/train_restormer.py`, `restormer/restormer_data.py` | `src/blur/finetune_deblur.py` (real pairs only) |
| `scripts/run_real_mixed.py`, `scripts/real_mixed_metrics.py`, `wiener/weiner_training.py` | `src/blur/tune_deblur.py` |
| `wiener/weiner_filter.py` (noise estimate, radius choice) | `src/blur/deblur_blocks/adaptive_wiener.py` |
| `src/deblur.py`, `src/deblur_blocks/`, `src/install_deblur_weights.py` | the same files, now in `src/blur/` |

Not ported: synthetic-blur training (no gain on real photos), MIDV / HDC2021 data preparation, directional and
higher-order Wiener, the bank router, edge sharpening, `handmade_evaluation.py` and the SSIM / CER scripts (replaced by
`box_ocr.py` and `evaluate.py`), PDF / HTML report generators. Images and fine-tuned weights were never in that git
repo; they exist only on the teammate's machine.
