# ECE253-ID-card-image-processing
ECE253 ID card image processing: out-of-focus blur and **glare** (specular highlight) removal.
It covers the blur part (`DEBLUR.md`), the glare part and both together.

## Blur-recovery blocks

See [DEBLUR.md](DEBLUR.md) for the independent Wiener and Restormer blocks:
`python src/deblur.py --method wiener|wiener_adaptive|restormer --in-dir DIR --out-dir DIR`,
and for tuning / training them on our splits (`src/tune_deblur.py`, `src/finetune_deblur.py`).
Earlier blur experiments are archived in `docs/blur_history/`.
They preserve rectified PNG filenames and dimensions and accept another block's
output directory as input. Results: "Deblurring" and "Joint deblur + deglare" below (`src/run_chain_experiments.sh`).

## Setup
```bash
python3.10 -m venv card_scanning_env && source card_scanning_env/bin/activate
pip install -r requirements.txt          # + torch torchvision for the deep-learning method
# pretrained DocSHRNet (code + weights, ~400 MB), used by src/finetune_dl.py
git clone --depth 1 https://github.com/shallweiwei/DocSHRNet third_party/DocSHRNet
mkdir -p third_party/weights && curl -L -o third_party/weights/docshrnet.pth \
  "https://drive.usercontent.google.com/download?id=1St86y5F_ltZXWGjIhjBfUkB81VsgRBvc&export=download&confirm=t"
```
Runs on an M2 MacBook (PyTorch MPS). All scripts are run from the project root.

## Pipeline
| step | script | output |
|---|---|---|
| rectify + align to the card's S_D reference | `src/rectify_ids.py` | `data/rectified/`, `alignment_log.csv` |
| pair glare images with references, card-level split | `src/prepare_data.py` | `out/pairs.csv`, `split.csv` |
| classical glare removal | `src/classical_glare.py` | `out/classical*/` |
| deep-learning glare removal (fine-tune + infer) | `src/finetune_dl.py` | `checkpoints/`, `out/dl_*/` |
| SSIM + OCR evaluation | `src/evaluate.py` | `results/{name}_per_image.csv`, `_summary.csv`, `all_methods.csv` |
| **main OCR metric:** fixed-box recognition per glare zone (PaddleOCR / Tesseract) | `src/box_ocr.py` | `results/box_layout.csv`, `results/box_{paddle,tesseract}/` |
| paired comparison with bootstrap CIs | `src/compare_runs.py` | `results/compare_*.csv` |
| external test set: MIDV-2020 highlight photos (rectify, align to scan, field boxes, glare masks) | `src/prepare_midv.py` | `data/midv/`, `out/pairs_midv.csv`, `results/box_layout_midv.csv` |
| runtime per image of each method | `src/benchmark_runtime.py` | `results/runtime_{test,midv}.csv` |
| OCR loss by glare zone (core / halo / outside) | `src/analyze_ocr_loss.py` | `results/ocr_loss_*.csv` |
| everything above after prepare_data | `bash src/run_glare_experiments.sh` | |

Split (by card, never by image): 14 train / 4 val / 4 test cards, giving 31 / 8 / 8 S_G and 30 / 8 / 9 B_G
usable glare images. Training and tuning use train (+ val). **Test is used only for the final numbers.**
OCR = PaddleOCR (PP-OCRv6), cached in `results/ocr_cache/`.

## Methods
### 1. Classical: two-zone highlight removal (`src/classical_glare.py`, `classical_params.json`)
The proposal's two-zone design, with a new veil estimator.
* **Model check.** A local linear fit of glare pixels against the reference gives
  gain + offset/255 ≈ 0.96 (median, 738 textured glare blocks). So the halo is a white veil
  `I = R(1-a) + 255a` (dichromatic model, white illuminant). It lifts the ink black level from ~20 to ~110
  (a ≈ 0.35-0.55), but the strokes are still there.
* **Seeds:** connected clipped blobs (min channel ≥ 248, ≥ 100 px). Images without seeds are returned unchanged
  (76 of 94 images have seeds).
* **Veil a(x) by "statistics inpainting"** (replaces the old `a = a0·exp(-d/tau)` guess):
  1. On a 16-px grid, measure the local ink black level and contrast.
  2. Interpolate the expected black level from textured blocks far from the seeds.
  3. `a = (observed - expected)/(255 - expected)` in textured blocks near the seeds, spread by normalised
     convolution, then grown by hysteresis from the seeds.
  4. Cap `a` by the lift of the white level, and apply a per-pixel background floor so blank paper is
     never darkened (this removed dark rings around cores).
  5. Invert the model on all three channels.
* **Core fill:** median colour of a ring around the core (`core_fill=flat`). Telea inpainting (`telea`) is kept as
  an ablation: it smears neighbouring strokes into the hole, and on train+val OCR it made word F1 worse on 7
  images and better on 1.
* **Parameter choice:**
  * Design choices were made on train+val with OCR and visual checks.
  * The numbers come from `--tune`: best ssim_glare with ssim_outside harm ≤ 0.002 on train+val S_G.
  * Tried and dropped: CLAHE in the halo (hurt OCR), and grayscale opening against laminate scratches
    (raised SSIM, not OCR).
  * Also dropped: **denoise before the veil inversion** (`--denoise`, ablation `out/classical_dn`).
    * Why it was tried: the inversion multiplies noise by 1/(1−a).
    * Variants: NLM, self-guided filter or bilateral, weighted by the veil; selected on train+val with the
      fixed-box metric.
    * NLM blurs thin strokes and hurts PaddleOCR. The guided filter (h = 8) was neutral for PaddleOCR and helped
      Tesseract on train S_G (+0.067).
    * On test it changed nothing for PaddleOCR, and for Tesseract halo it was −0.10 vs classical (n.s.).
      The gain did not transfer.

### 2. Deep learning: detection + removal network (`src/finetune_dl.py`)
The proposal's JSHDR network (Fu et al., CVPR 2021) is no longer available: its GitHub repository and weights are
gone. We use **DocSHRNet** (Xu et al., PRCV 2025), a NAFNet-style U-Net (34.6 M parameters) pretrained on
DocHighlight, a dataset of real *document* specular highlights. We turn it into a multi-task
detection-and-removal network:
* **Removal head:** the pretrained output, `y = x + head(features)`.
* **Detection head (new):** 4 dilated convolutions on the shared last decoder features, giving a glare probability `m`.
* **Output:** `m·y + (1-m)·x`. The network only changes pixels it detects as glare, so the rest of the card is
  untouched by design.
* **Training data:** the 31 train S_G pairs, cropped to 256 px (70% of crops centred on glare), with flips.
* **Loss:** glare-weighted L1 + 0.2·(1-SSIM) + 0.1·BCE(m, glare region from the reference).
* **Optimiser:** AdamW, lr 5e-5 (backbone) / 5e-4 (head), 1500 iterations (~1.5 h on M2 MPS).
* **Checkpoint selection:** on val S_G with the do-no-harm rule. Best was iteration 1350.

## Results (test split; 4 unseen cards)

### Main OCR metric: fixed-box recognition per glare zone (`src/box_ocr.py`)
Whole-card OCR mixes in text-detector churn (boxes split / merge differently on near-identical images) and
dilutes the glare (~10% of the card). So:
* **Fixed boxes.** The line boxes PaddleOCR detects on each card's clean S_D reference are used for every image
  of that card (all images are aligned to it). 342 of 368 boxes, ~17 per card, are labelled with their true text:
  the closest substring of `ground_truth/{card}.txt` to the reference reading. Dropped: field numbers (< 3 chars),
  signatures, non-Latin text.
* **Recognition only.** Each box is cropped (no padding: lines are tightly packed; vertical boxes are rotated as
  in the PaddleOCR pipeline) and recognised:
  * PaddleOCR `TextRecognition` (PP-OCRv6_medium_rec, the pipeline's own recogniser);
  * Tesseract `--psm 7` (one line, crop upscaled 2×).

  A box whose pixels a method does not change gets a bit-identical reading.
* **Zones** per image from the untreated image: core (clipped) / halo (rest of the glare region) / outside.
* **Score:** character accuracy = 1 − edit distance / target length, on lowercase letters and digits, pooled over
  the boxes of a zone. `digits` = boxes with ≥ 4 digits (dates, ID numbers).
* **Noise floor** (`ref_jitter`): the reference with every box shifted by ≤ 2 px, noise σ = 2 and JPEG q = 90.
  Its paired difference from `reference` is ≤ 0.006 per zone for PaddleOCR, and up to ~0.03-0.07 for Tesseract.
  (A second real clean photo per card would be the proper floor; see NEXT_STEPS.)

**Test S_G (n = 8 images; 5 core / 26 halo / 77 outside / 18 digit-in-glare boxes), character accuracy:**
| method | PaddleOCR core | halo | outside | digits in glare | Tesseract core | halo | outside | digits in glare |
|---|---|---|---|---|---|---|---|---|
| untreated | 0.444 | 0.869 | 0.940 | 0.855 | 0.133 | 0.417 | 0.744 | 0.379 |
| classical | 0.378 | 0.854 | 0.939 | 0.844 | 0.178 | 0.480 | 0.747 | 0.441 |
| DL zero-shot | 0.400 | 0.714 | 0.939 | 0.754 | 0.222 | 0.574 | 0.763 | 0.539 |
| DL fine-tuned | **0.467** | 0.817 | 0.937 | 0.812 | **0.244** | **0.620** | **0.779** | **0.586** |
| *reference (ceiling)* | 0.978 | 0.994 | 0.987 | 0.988 | 0.867 | 0.837 | 0.848 | 0.770 |

**Paired differences vs untreated, test S_G halo** (95% bootstrap CI over images, images better/worse;
`results/box_*/compare_untreated_vs_*.csv`):
| method | PaddleOCR halo | Tesseract halo | Tesseract digits in glare |
|---|---|---|---|
| classical | −0.016 [−0.035, −0.003] 0/3 | +0.100 n.s. 5/2 | +0.083 n.s. |
| DL zero-shot | −0.140 [−0.216, −0.063] 0/5 | +0.163 [0.046, 0.310] 6/0 | +0.178 [0.019, 0.337] |
| DL fine-tuned | −0.043 [−0.083, −0.011] 0/3 | **+0.217 [0.106, 0.363] 7/0** | **+0.223 [0.085, 0.361]** |

The same signs hold on train and val, which were not used to pick anything here:
* PaddleOCR halo: DL fine-tuned −0.016 on train S_G (significant); classical +0.006 on train.
* Tesseract halo: DL fine-tuned +0.16 on train S_G and +0.17 on val S_G; zero-shot +0.12 / +0.07.

B_G (blur + glare): the halo is unreadable for both engines (char accuracy ≤ 0.25), and no method changes that
significantly on test.

### Whole-card OCR and SSIM
Columns: SSIM to the glare-free reference (whole image / inside the glare / far from the glare);
CER and word F1 of the OCR text versus the OCR text of the reference; F1 versus the typed ground truth.

**S_G (sharp + glare, n = 8)**
| method | ssim | ssim_glare | ssim_outside | CER ↓ | word_f1 | f1_gt |
|---|---|---|---|---|---|---|
| untreated | 0.696 | 0.511 | 0.715 | 0.125 | 0.757 | 0.753 |
| classical (final) | 0.694 | 0.527 | 0.712 | 0.130 | 0.725 | 0.737 |
| — veil only | 0.693 | 0.512 | 0.712 | 0.129 | 0.725 | 0.737 |
| — core fill only | 0.697 | 0.524 | 0.715 | 0.125 | 0.754 | 0.757 |
| — Telea core fill | 0.693 | 0.524 | 0.712 | 0.130 | 0.728 | 0.735 |
| — old decay veil (before modifications) | 0.686 | 0.518 | 0.705 | 0.107 | 0.743 | 0.746 |
| DL zero-shot (DocSHRNet) | **0.748** | 0.545 | **0.766** | 0.135 | 0.742 | 0.756 |
| DL fine-tuned (multi-task) | 0.730 | **0.624** | 0.734 | **0.109** | 0.756 | **0.769** |

**B_G (blur + glare, n = 9):** no method changes OCR significantly.
f1_gt: untreated 0.390, classical 0.392, DL zero-shot 0.374, DL fine-tuned 0.364.
Blur dominates; the DL model was trained only on sharp images.

**Tesseract, whole card** (`--psm 11`; runs `*_tess`, `results/compare_untreated_tess_vs_*.csv`).
Tesseract reads whole ID cards poorly: S_G f1_gt is 0.40 untreated vs 0.75 with PaddleOCR (layout, not glare).
The line crops above avoid this floor.
* S_G f1_gt: untreated 0.399, classical 0.407, DL zero-shot 0.404, DL fine-tuned 0.429.
* The only significant paired change is DL fine-tuned recall_gt, +0.052 [0.009, 0.098] (5/1).

**Paired differences on test S_G** (`results/compare_*.csv`, 95% bootstrap CI, images better/worse):
| comparison | ssim_glare | ssim_outside | CER (+ = better) | f1_gt |
|---|---|---|---|---|
| classical vs untreated | +0.016 [0.003, 0.030] 7/1 | −0.003 [−0.005, −0.001] 0/7 | −0.005 [−0.010, −0.001] 0/3 | −0.017 n.s. |
| DL fine-tuned vs untreated | **+0.113 [0.059, 0.169] 8/0** | **+0.019 [0.005, 0.034] 8/0** | +0.016 n.s. 5/1 | +0.016 n.s. |
| DL fine-tuned vs zero-shot | +0.080 [0.007, 0.157] | −0.032 (gate keeps the input) | **+0.026 [0.005, 0.048] 6/1** | +0.014 n.s. |
| DL fine-tuned vs classical | **+0.097 [0.041, 0.159] 8/0** | +0.022 [0.007, 0.038] 8/0 | +0.020 n.s. | +0.033 n.s. |

**OCR word loss by zone** (all 47 S_G images, `results/ocr_loss_*.csv`):
| | core (38 words) | halo (490) | halo, digits only (138) | outside (1392) |
|---|---|---|---|---|
| untreated | 0.82 | 0.44 | 0.50 | 0.11 |
| classical | 0.84 | 0.44 | 0.51 | 0.12 |
| DL zero-shot | 0.79 | 0.45 | 0.47 | 0.11 |
| DL fine-tuned | 0.84 | 0.42 | 0.44 | 0.11 |

**Glare detection IoU** with the reference-derived glare region (`results/detection_iou.csv`), test S_G:
DL detection head 0.30 (0.59 on train: it overfits with 31 images); classical seeds+veil region 0.27
(it acts only near clipped seeds). The two masks agree with each other at IoU 0.44.

Figures: `results/figures/test_S_G_comparison.jpg` (whole cards) and `results/figures/test_S_G_zoom.png`
(strongest glare). Classical debug panels (input | veil map + cores | output) are in `out/classical/debug/`.

### External test set: MIDV-2020 (`src/prepare_midv.py`; `results/box_{paddle,tesseract}_midv/`)
[MIDV-2020](https://arxiv.org/abs/2107.00396) (CC BY-SA 2.5) has 1000 mock identity documents of 10 types
(5 ID cards, 5 passports), each with a digital template (field rectangles + text), a flatbed scan and a phone
photo. Photos 60–69 of every type are labelled "highlight present": **100 real glare photos of 100 different
documents, from two phones (iPhone XR, Samsung S10), none used for tuning or training.** Download from the public
mirror `ftp://smartengines.com/midv-2020/dataset/` (`photo.tar`, `templates.tar`, `scan_upright.tar`; ~6 GB;
commands in the script's docstring).
* **Rectification.** Photo and scan are warped from their annotated document quads to our resolution
  (~11.8 px/mm: cards 1012 px wide, passports 1478 px). The photo is then ECC-aligned to the scan; the ECC step is
  kept only if it raises the correlation and moves no corner > 15 px (69/100), otherwise the quad warp alone.
* **Reference = the scan** of the same printed document (glare-free). For the glare zones it is tone-matched to the
  photo (per-channel gain/offset + smooth illumination field, σ = 150 px, fitted outside outliers), then the usual
  rule: > 40 grey levels brighter. Median glare area 10.5% of the document.
* **Fixed boxes = annotated fields** of the template, mapped to the scan frame (template→scan ECC, same guard).
  997 of 1341 fields are kept: Latin letters/digits only after folding accents (drops Greek, Cyrillic, Ə),
  ≥ 3 characters, horizontal, not overlapping another field. Crops get a 2 px pad (the field rectangles are
  tight and fields are well separated); `--fold-accents` strips diacritics from the readings too.
* **Same frozen methods**: `classical_params.json`, DocSHRNet zero-shot, `checkpoints/docshr_mt_best.pt`.
  Scored against the annotated field text only.

**MIDV-2020 highlight photos (n = 100; 49 core / 434 halo / 514 outside / 232 digit-in-glare boxes), character accuracy:**
| method | PaddleOCR core | halo | outside | digits in glare | Tesseract core | halo | outside | digits in glare |
|---|---|---|---|---|---|---|---|---|
| untreated | **0.636** | **0.830** | **0.962** | **0.838** | 0.218 | 0.334 | 0.747 | 0.333 |
| classical | 0.484 | 0.829 | 0.962 | 0.821 | 0.159 | 0.333 | 0.748 | 0.333 |
| DL zero-shot | 0.467 | 0.735 | 0.957 | 0.730 | 0.237 | 0.435 | **0.785** | 0.431 |
| DL fine-tuned | 0.511 | 0.754 | 0.952 | 0.743 | **0.281** | **0.447** | 0.776 | **0.446** |
| *reference = scan (ceiling)* | 1.000 | 0.995 | 0.993 | 0.994 | 0.922 | 0.900 | 0.890 | 0.944 |

**Paired differences vs untreated** (95% bootstrap CI over images, images better/worse; per-image scores pool the
boxes of the zone, so n = images with ≥ 1 box in it: 91 halo, 16 core):
| method | PaddleOCR halo | PaddleOCR core | Tesseract halo | Tesseract digits in glare |
|---|---|---|---|---|
| classical | −0.001 [−0.003, −0.000] 1/8 | −0.105 [−0.174, −0.041] 2/10 | +0.002 n.s. 7/10 | +0.003 n.s. |
| DL zero-shot | −0.105 [−0.131, −0.081] 5/68 | −0.160 [−0.248, −0.080] | +0.114 [0.081, 0.149] 61/10 | +0.107 [0.072, 0.142] |
| DL fine-tuned | −0.086 [−0.105, −0.068] 3/68 | −0.133 [−0.219, −0.059] | **+0.125 [0.094, 0.159] 70/10** | **+0.126 [0.095, 0.160]** |

* Noise floor: `ref_jitter` vs `reference` differs by ≤ 0.002 per zone for PaddleOCR and ≤ 0.03 for Tesseract.
* DL fine-tuned vs zero-shot: not significantly different in the halo (PaddleOCR +0.020, Tesseract +0.012, both
  n.s.); fine-tuned is better in the PaddleOCR core (+0.027 [0.002, 0.053]).
* The signs are the same for all 10 document types and both phones (`results/midv_by_doctype.csv`): DL raises
  Tesseract glare-zone accuracy for 9–10 of 10 types and lowers PaddleOCR's for all 10.
* Figure: `results/figures/midv_examples.jpg` (untreated / classical / DL zero-shot / DL fine-tuned / scan).

### Multi-frame fusion: MIDV-2020 clips (`src/prepare_midv_clips.py`, `src/fuse_frames.py`; `results/box_{paddle,tesseract}_midv_clips/`)
Glare moves with the camera; the document does not. MIDV-2020 clips 90–99 of every type are "highlight present"
videos: **100 clips of 100 documents** (not the photo documents), every 6th frame annotated with the document quad
(4K frames, document ~1,530 px wide). Nothing is learned or tuned on them; all constants were fixed before the first
OCR run. Download: `clips.tar` from the same mirror (10.3 GB; only the frames of clips 90–99 are extracted).
* **Frames.** Each frame is warped from its quad. Dropped if < 80% of the document is in the image or it is
  motion-blurred (variance of Laplacian < 0.5 × clip median; blur is bimodal, ~0.1 vs ~1 × median). Pixels outside
  the image are tracked per frame and treated as missing (14% of frames have one edge cut off).
* **Alignment.** Anchor = middle sharp frame showing the whole document, aligned to the scan exactly like the photos
  (ECC kept for 65/100). Every other frame → anchor by a homography: SIFT + RANSAC initial guess, refined by ECC on
  locally normalised images with glare masked in both frames (plain ECC failed: the glare veil dominates the
  correlation). Kept only if the correlation rises, no corner moves > 80 px, and ≥ 5 clean tiles agree to ≤ 1 px
  (phase correlation). **4,641 of 6,258 frames kept** (median 52 per clip; residual median 0.44 px, p90 0.80 px).
  Drops: 1,156 residual (mostly bent passport pages, 24% of passport frames vs 12% of ID-card frames), 242 blur,
  219 ECC. A non-rigid correction was tried and rejected: on held-out quadrants a polynomial displacement field
  extrapolated worse than the homography alone, and fusion replaces pixels exactly where nothing can be measured.
  Each frame is stored as one homography from the raw frame (`out/midv_clips_frames.csv`), so it is interpolated once.
* **Key frame = the "untreated" image**, two rules (= two "splits"): **keymid** = the anchor (a typical capture);
  **keymax** = the frame with the most clipped core inside the field boxes (where recovery matters). Same frame in 7 clips.
  Zones (core / halo / outside) come from the key frame, as for the photos; 994 labelled field boxes.
* **Fusion** (main): in each frame a pixel is glare if clipped or > 30 grey levels above its glare-free level
  (25th percentile over time; the median is often glare itself, as the glare covers some spots in > half the frames).
  **Only the key frame's glare pixels are replaced**, by the mean over the frames where that pixel is clean; all other
  pixels are untouched. **Ablation:** plain temporal median. **Baselines:** the best single frame (least glare in the
  field boxes) and the **best frame per field** (each field from the frame where it has the least glare: an automatic
  per-field retake). None of them uses the reference.
* **Recoverable fraction:** in 99/100 clips every clipped-core pixel of the key frame is unclipped in ≥ 1 other aligned
  frame (strict per-pixel check on 12 clips: typically in almost all of them). The exception has 1 usable frame.

**PaddleOCR, character accuracy per zone** (boxes: keymid 172 core / 269 halo / 553 outside; keymax 492 / 156 / 346):
| method | keymid core | halo | outside | keymax core | halo | outside |
|---|---|---|---|---|---|---|
| untreated (key frame) | 0.524 | 0.822 | 0.952 | 0.479 | 0.847 | 0.950 |
| classical | 0.461 | 0.809 | 0.949 | 0.406 | 0.829 | 0.949 |
| DL zero-shot | 0.438 | 0.759 | 0.927 | 0.389 | 0.801 | 0.936 |
| DL fine-tuned | 0.482 | 0.777 | 0.939 | 0.424 | 0.812 | 0.950 |
| best single frame | 0.943 | 0.920 | 0.963 | 0.960 | 0.907 | 0.942 |
| best frame per field | 0.981 | 0.946 | 0.977 | 0.977 | 0.942 | 0.964 |
| temporal median (ablation) | 0.963 | 0.930 | 0.980 | 0.970 | 0.941 | 0.961 |
| **fused** | **0.988** | **0.951** | **0.981** | **0.989** | **0.947** | **0.967** |
| *reference = scan (ceiling)* | 0.997 | 0.992 | 0.994 | 0.996 | 0.994 | 0.991 |

**Paired differences, PaddleOCR** (95% bootstrap CI over clips; clips better/worse; n = clips with ≥ 1 box in the zone):
| comparison | keymax core (n = 97) | keymax halo (57) | keymax all fields (100) | keymid core (57) | keymid halo (73) | keymid all (100) |
|---|---|---|---|---|---|---|
| fused vs untreated | **+0.505 [0.460, 0.550] 95/0** | +0.081 [0.052, 0.113] 27/0 | +0.324 [0.285, 0.363] | **+0.482 [0.415, 0.548] 53/0** | +0.131 [0.097, 0.167] 49/1 | +0.150 [0.119, 0.182] |
| fused vs best single frame | +0.030 [0.015, 0.050] 28/4 | +0.051 [0.014, 0.098] | +0.026 [0.014, 0.041] | +0.064 [0.025, 0.111] 17/3 | +0.032 [0.011, 0.065] | +0.024 [0.014, 0.035] |
| fused vs best frame per field | +0.010 [0.005, 0.016] 21/7 | +0.001 n.s. | +0.006 [0.003, 0.010] 26/6 | +0.004 n.s. | +0.003 n.s. | +0.004 [0.002, 0.007] 23/8 |
| DL fine-tuned vs untreated | −0.079 [−0.106, −0.053] | −0.040 [−0.065, −0.019] | −0.052 | −0.051 [−0.094, −0.008] | −0.062 [−0.092, −0.035] | −0.024 |
| classical vs untreated | −0.111 [−0.141, −0.084] | −0.025 [−0.041, −0.012] | −0.073 | −0.075 [−0.106, −0.044] | −0.026 [−0.050, −0.009] | −0.021 |

**Tesseract, character accuracy per zone** (same boxes):
| method | keymid core | halo | outside | keymax core | halo | outside |
|---|---|---|---|---|---|---|
| untreated (key frame) | 0.222 | 0.455 | 0.778 | 0.183 | 0.482 | 0.789 |
| classical | 0.244 | 0.479 | 0.781 | 0.210 | 0.471 | 0.795 |
| DL zero-shot | 0.261 | 0.529 | 0.807 | 0.238 | 0.547 | 0.811 |
| DL fine-tuned | 0.281 | 0.553 | 0.800 | 0.266 | 0.584 | 0.812 |
| best single frame | 0.796 | 0.734 | 0.825 | 0.806 | 0.719 | 0.808 |
| best frame per field | 0.851 | **0.775** | **0.838** | 0.831 | 0.771 | **0.833** |
| temporal median (ablation) | 0.737 | 0.731 | 0.837 | 0.782 | 0.740 | **0.834** |
| **fused** | **0.871** | 0.769 | **0.838** | **0.856** | **0.789** | 0.831 |
| *reference = scan (ceiling)* | 0.887 | 0.880 | 0.908 | 0.902 | 0.897 | 0.887 |

**Paired differences, Tesseract:**
| comparison | keymax core (n = 97) | keymax halo (57) | keymax all fields (100) | keymid core (57) | keymid halo (73) | keymid all (100) |
|---|---|---|---|---|---|---|
| fused vs untreated | **+0.720 [0.670, 0.768] 95/1** | +0.303 [0.219, 0.389] 39/4 | +0.488 [0.440, 0.535] | **+0.689 [0.615, 0.761] 54/0** | +0.420 [0.347, 0.494] 61/3 | +0.269 [0.221, 0.320] |
| fused vs best single frame | +0.058 [0.029, 0.090] 47/14 | +0.078 [0.027, 0.136] | +0.041 [0.023, 0.058] | +0.097 [0.046, 0.156] | +0.052 [0.014, 0.095] | +0.024 [0.006, 0.042] |
| fused vs best frame per field | +0.030 [0.010, 0.050] 48/16 | +0.020 n.s. | +0.013 [0.001, 0.023] 53/19 | +0.022 n.s. | +0.012 n.s. | +0.004 n.s. |
| DL fine-tuned vs untreated | +0.077 [0.053, 0.101] | +0.102 [0.061, 0.148] | +0.062 | +0.061 [0.024, 0.098] | +0.137 [0.092, 0.185] | +0.060 |
| classical vs untreated | +0.018 n.s. | −0.012 n.s. | +0.007 n.s. | +0.024 n.s. | +0.042 [0.010, 0.074] | +0.017 |

* **Clipped text comes back, for both engines**: PaddleOCR core 0.48–0.52 → 0.99 (ceiling 0.996), Tesseract
  0.18–0.22 → 0.86–0.87 (ceiling 0.89–0.90); at most 1 clip worse per zone and engine. The best single-image method gains
  ≤ +0.08 in the core for Tesseract and makes it worse for PaddleOCR (the text is physically gone in one frame).
* **The engine dependence of the single-image methods replicates on the clips**: DL fine-tuned helps Tesseract
  (halo +0.10 / +0.14) and hurts PaddleOCR (halo −0.04 / −0.06), as on our test split and the MIDV photos.
  Fusion helps both engines.
* **Fusion vs the automatic retake**: fused beats the best single frame clearly (one frame trades glare in one field
  for glare in another) and the best frame per field only slightly: over all fields PaddleOCR +0.006 / +0.004,
  Tesseract +0.013 (keymax) / n.s. (keymid); in the core +0.010 / +0.030 on keymax. An app that re-reads each field
  from its cleanest frame gets almost all of the gain; fusion adds a little, mostly on fields that are partly
  glared in every frame.
* **Replacing only the glare pixels matters**: the temporal median is below fused in every zone (blur from
  residual misalignment and averaging), and its outside-zone gain is not significant.
* Fused's small "outside" gain comes only from the 188 outside boxes its own glare test flagged (weak glare
  below the 40-level zone threshold: 0.877 → 0.982, 48 better / 3 worse); the other 711 outside boxes are untouched.
* Noise floor: `ref_jitter` vs `reference` differs by ≤ 0.005 per zone for PaddleOCR and ≤ 0.02 for Tesseract.
* Runtime (M2, CPU): alignment ~1 s per frame; fusion 4.3 s per clip once the ~50 aligned frames are loaded (1.8 s).
* Figure: `results/figures/midv_clips_fusion.jpg` (key frame | fused | best single frame | best frame per field | scan).

### Deblurring: B_D (`src/deblur.py`, `src/tune_deblur.py`, `src/finetune_deblur.py`; `results/box_*_blur/`, `results/chain/blur_*`)
All settings were chosen on train + val B_D (Wiener, adaptive Wiener) or trained on train and selected on val
(Restormer: step 1000 of 2000), then frozen: `deblur_wiener_params.json` (radius 6 / K 0.03),
`deblur_wiener_adaptive_params.json`, `checkpoints/deblur/restormer_ours.pt`. No B_G image was used.
**Test B_D (n = 4 images, 54 boxes)**: character accuracy pooled over all boxes, and paired changes vs untreated
(mean of per-image changes [95% bootstrap CI] images better/worse; they differ slightly from pooled differences):
| method | PaddleOCR | Tesseract | PaddleOCR change | Tesseract change | SSIM change |
|---|---|---|---|---|---|
| untreated | 0.782 | 0.307 | – | – | – |
| Wiener, old setting (r 3 / K 0.3) | 0.754 | 0.284 | −0.024 [−0.046, −0.005] 0/3 | −0.010 n.s. | −0.002 n.s. |
| Wiener, retuned | 0.568 | 0.284 | −0.210 n.s. 1/2 | −0.084 n.s. | −0.042 n.s. |
| Wiener, adaptive | 0.530 | 0.242 | −0.236 n.s. 1/2 | −0.070 n.s. | −0.012 n.s. |
| Restormer, pretrained | 0.809 | 0.388 | +0.011 n.s. 2/2 | +0.071 [0.000, 0.126] 3/1 | +0.025 [0.017, 0.034] 4/0 |
| Restormer, fine-tuned | **0.838** | **0.745** | +0.026 [−0.103, 0.150] n.s. 2/1 | **+0.376 [0.233, 0.510] 4/0** | **+0.074 [0.040, 0.100] 4/0** |
| *reference (ceiling)* | 0.989 | 0.846 | | | |

* n = 4 test images. On the 20 train + val images the fine-tuned Restormer gains +0.153 (PaddleOCR) and +0.502
  (Tesseract), but it was trained on 16 of them; the two val designs F07 and F13 also occur in train, the four
  test designs do not.
* Wiener looked useful on train + val (Tesseract +0.177 retuned, +0.108 adaptive, both tuned on these images) and did
  not transfer: B03 drops from 0.97 to 0.21 (PaddleOCR). The adaptive radius does not transfer either.

### Joint deblur + deglare: B_G (`src/run_chain_experiments.sh`, `src/summarise_chain.py`; `results/box_*_chain/`, `results/chain/`)
Chains are named in processing order; all parts are the frozen single methods (DL glare was trained on sharp S_G only).
**Test B_G (n = 9), all boxes:**
| method | PaddleOCR | Tesseract | PaddleOCR change vs untreated | Tesseract change | SSIM change |
|---|---|---|---|---|---|
| untreated | 0.550 | 0.189 | – | – | – |
| classical | 0.546 | 0.212 | −0.003 n.s. | +0.033 [0.004, 0.067] 6/1 | −0.003 [−0.006, −0.001] |
| DL ft | 0.550 | 0.222 | +0.003 n.s. | +0.051 [0.001, 0.130] 8/1 | +0.004 n.s. |
| Wiener / adaptive Wiener | 0.426 / 0.470 | 0.197 / 0.158 | −0.121 / −0.078 n.s. | −0.061 / −0.013 n.s. | −0.038* / −0.004 n.s. |
| Wiener → classical / classical → Wiener | 0.422 / 0.434 | 0.201 / 0.204 | −0.124 / −0.117 n.s. | −0.025 / −0.022 n.s. | −0.044* / −0.043* |
| Restormer pretrained | 0.572 | 0.258 | +0.020 n.s. | +0.064 [0.034, 0.095] 8/1 | +0.024* |
| Restormer ft | 0.669 | 0.532 | +0.087 [0.006, 0.175] 5/2 | +0.294 [0.169, 0.426] 8/1 | +0.070* |
| Restormer ft → classical | 0.677 | 0.536 | +0.094 [0.009, 0.187] 5/2 | +0.316 [0.206, 0.422] 9/0 | +0.067* |
| **Restormer ft → DL ft** | **0.679** | **0.559** | +0.090 [0.006, 0.179] 4/2 | **+0.339 [0.223, 0.454] 9/0** | **+0.074*** |
| DL ft → Restormer ft | 0.676 | 0.545 | +0.094 [0.009, 0.186] 5/3 | +0.332 [0.228, 0.433] 9/0 | +0.071* |
| *reference (ceiling)* | 0.989 | 0.852 | | | |

(* = CI excludes 0.)
* **Deblurring does most of the work.** Adding DL deglare after Restormer changes PaddleOCR by +0.003 n.s. and
  Tesseract by +0.045 [0.008, 0.088]. Neither glare method alone changes PaddleOCR on B_G.
* **Order:** no significant difference on test between the two learned orders (PaddleOCR −0.004, Tesseract +0.007 for
  deblur first). Clipped (core) text stays lost in every run.
* About 70% of PaddleOCR's gap to the reference remains (0.679 vs 0.989), and about 45% of Tesseract's.
* **Do no harm, S_G (n = 8 sharp images):** Restormer ft PaddleOCR −0.000, Tesseract +0.036 [0.010, 0.064]; the learned
  chains cost PaddleOCR −0.012 to −0.017 (as DL deglare alone, −0.015). The fixed Wiener filter destroys sharp text
  (−0.47 / −0.44); the adaptive one costs −0.007 / −0.018 n.s.

**MIDV-2020 photos (n = 100, external, nothing tuned on them), all boxes, paired change vs untreated** (`results/box_*_midv_chain/`):
| method | PaddleOCR | change | Tesseract | change |
|---|---|---|---|---|
| untreated | 0.887 | – | 0.534 | – |
| classical | 0.880 | −0.008 [−0.016, −0.002] | 0.532 | +0.000 n.s. |
| DL ft | 0.842 | −0.052 [−0.067, −0.040] | 0.603 | +0.074 [0.055, 0.094] |
| Wiener | 0.386 | −0.465 [−0.496, −0.432] | 0.043 | −0.476 [−0.523, −0.429] |
| adaptive Wiener | 0.856 | −0.046 [−0.068, −0.027] | 0.502 | −0.058 [−0.092, −0.030] |
| adaptive Wiener → classical / classical → adaptive Wiener | 0.849 / 0.847 | −0.054 / −0.057 | 0.500 / 0.496 | −0.059 / −0.062 |
| Restormer ft | 0.884 | −0.005 [−0.008, −0.001] | 0.545 | +0.011 [0.001, 0.023] |
| **Restormer ft → DL ft** | 0.838 | −0.059 [−0.077, −0.043] | **0.614** | **+0.086 [0.066, 0.107] 81/10** |
| DL ft → Restormer ft | 0.842 | −0.056 [−0.071, −0.042] | 0.595 | +0.067 [0.047, 0.087] |
| *reference = scan* | 0.994 | | 0.896 | |

* These photos were selected for glare, not blur, so the deblur step has little to restore: Restormer is close to neutral.
  Nearly all of the chain's gain (Tesseract) and loss (PaddleOCR) is the DL deglare step.
* Deblur first beats deglare first for Tesseract (+0.019 [0.007, 0.032]; PaddleOCR n.s.) and beats DL alone by only
  +0.012 [0.000, 0.024].
* Wiener hurts both engines (its setting was chosen on blurred images; these are mostly not blurred). Adding classical deglare
  does not recover it.

**MIDV-2020 clips: deblurring the fused image** (`results/box_*_midv_clips_chain/`; 100 clips, keymax / keymid, all boxes; change vs fused):
| method | PaddleOCR | change | Tesseract | change |
|---|---|---|---|---|
| fused | 0.976 / 0.974 | – | 0.838 / 0.824 | – |
| fused → Wiener | 0.500 / 0.483 | −0.409 / −0.426 | 0.088 / 0.082 | −0.729 / −0.723 |
| fused → adaptive Wiener | 0.954 / 0.951 | −0.035 / −0.029 | 0.703 / 0.694 | −0.167 / −0.159 |
| fused → Restormer ft | 0.976 / 0.973 | +0.001 / −0.002 n.s. | 0.832 / 0.818 | +0.000 / −0.002 n.s. |

* Fusion already reaches the ceiling within 0.02 (PaddleOCR) and 0.06–0.07 (Tesseract): deblurring after it does not help.
  The single-image chains were not run on clips (the key frames pass the blur filter, and single-image deglare cannot
  restore clipped text), nor was deblurring single frames before fusion (4% of the frames are dropped for blur).

### Runtime (`src/benchmark_runtime.py`, `results/runtime_test.csv`)
Processing call only (image already in memory), median over our 17 test images (1012×638), M2 MacBook,
torch 2.14 (4 CPU threads). Each image was timed 3× (DL on CPU: 1×, since single calls vary < 1%):
| method | device | s / image |
|---|---|---|
| classical | CPU | **0.086** |
| DL zero-shot | MPS (M2 GPU) | 2.01 |
| DL fine-tuned | MPS | 2.37 |
| DL zero-shot | CPU | 9.49 |
| DL fine-tuned | CPU | 9.62 |
| Wiener | CPU | 0.049 |
| adaptive Wiener (with its 9 Tesseract calls) | CPU | 2.72 |
| Restormer ft | MPS | 19.6 |

* The classical method is ~25× faster than DL on the GPU and ~110× faster on the CPU.
* It costs ~0.01 s on images with no clipped seed, because it returns the image unchanged
  (MIDV: 15 of 20 sampled photos).
* DL time grows with the pixel count: 4.8 s (zero-shot) and 5.9 s (fine-tuned) per MIDV passport (1478×1040) on MPS
  (`results/runtime_midv.csv`, 20 photos, 1 repeat).
* Fine-tuned is slightly slower than zero-shot on MPS because of the added detection head.
* Deblur rows: `results/runtime_deblur_test.csv`, same protocol. Restormer (512 px tiles) is the slow step: Restormer → DL takes ~22 s per image on MPS. Not timed on CPU.

### Findings
**With several frames, text under clipped glare comes back** (100 MIDV-2020 clips, nothing tuned on them): fusing
aligned video frames raises core (clipped) character accuracy from ~0.5 to 0.99 for PaddleOCR and from ~0.2 to 0.86
for Tesseract, close to the scan ceiling, where the single-image methods below gain at most +0.08 or make it worse.
Most of the gain is already available by reading each field from its cleanest frame (fusion adds +0.004–0.013 over
all fields, n.s. for Tesseract on keymid). For an app: capture a short burst, not one photo.

0. **The main result replicates on 100 external photos** (MIDV-2020, other documents, phones and glare; nothing tuned
   on it): DL helps Tesseract (halo +0.125, 70/10 images) and hurts PaddleOCR (halo −0.086, 3/68); classical is
   neutral in the halo and its flat core fill removes core text PaddleOCR could still partly read (core −0.105).
   Unlike on our test split, fine-tuning gives no significant OCR gain over zero-shot there: the MIDV glare is
   mostly textured laminate / hologram reflection, unlike our 31 training images.
1. **Whether glare removal helps OCR depends on the OCR engine** (fixed-box metric, consistent on
   train, val and test).
   * **Tesseract** (binarises the line) loses most of the halo text to the veil: halo character accuracy is 0.42,
     vs 0.84 on the clean reference. DL fine-tuned recovers about half the gap (+0.22 on test, 7/0 images;
     digits in glare +0.22). Zero-shot recovers less (+0.16). Classical gains +0.10, not significant.
   * **PaddleOCR**'s neural recogniser already reads 87% of halo characters untreated (99% on the reference).
     There every method makes the halo slightly *worse*: DL fine-tuned −0.04, zero-shot −0.14, classical −0.02.
     The processed strokes (smoothed, or with invented texture) confuse it more than the veil does.
   * Core text (clipped) is lost for every engine and method; the few core boxes make the core numbers noisy.
   * The earlier whole-card result ("no significant change") was mostly detection churn plus dilution.
     With fixed boxes the PaddleOCR noise floor is ≤ 0.006, so these effects are real.
   * The strongest glare is not a smooth veil. It is a textured reflection of the laminate (scratches,
     hologram) in which the strokes are physically gone (see the zoom figure).
   * Practical reading: a preprocessing step for a classical / binarising OCR engine should use DL fine-tuned.
     A modern neural recogniser should get the untreated image, plus a "recapture" flag for clipped fields.
2. **SSIM and OCR disagree.**
   * The DL model is far better by SSIM, inside and outside the glare. Zero-shot raises ssim_outside the most
     simply by smoothing the whole card (cf. `ssim_diagnose.py`: light blur raises SSIM from 0.76 to 0.92).
   * It does not read better for PaddleOCR. Zero-shot, with the best whole-image SSIM, is the worst method
     for PaddleOCR halo text.
   * Report both, and treat SSIM gains without OCR gains with care.
3. **Fine-tuning matters for the network.**
   * Zero-shot DocSHRNet paints dark text-like smears into clipped cores (hallucination), and its CER is worse
     than untreated (0.135 vs 0.125).
   * After fine-tuning, cores become smooth fills and CER is significantly better than zero-shot (0.109).
   * The gate keeps text outside the glare unchanged.
4. **The classical method is safe but weak** (fixed-box metric, test halo: PaddleOCR −0.016,
   Tesseract +0.10 n.s.; denoising before the inversion does not change that).
   * The veil model is physically right for the halo, and the new estimator removes the harm of the old
     distance-decay veil away from the glare (ssim_outside significantly better than decay on val and test).
   * On the other hand, the decay veil had the higher ssim_glare on val, where its parameters were tuned;
     on test the two are not significantly different.
   * But it acts only near clipped seeds (18 of 94 images untouched) and cannot undo textured specular
     reflection.
   * Inpainting the core does not help OCR. The information there is lost; flat fill at least does not invent
     strokes. Large cores should rather be flagged for recapture, as in the proposal.

Caveats:
* MIDV-2020: the reference is a scan, not a photo, so its glare masks depend on the tone matching (they still
  include a few edges of faces / holograms), and only the Latin fields are scored. SSIM is not reported there
  (different capture device).
* Test has 8 S_G / 9 B_G images from 4 cards. The CIs are wide, and several choices that looked better on
  train+val did not transfer.
* The DL glare masks used for training come from reference differences (> 40 grey levels), so very weak veils
  are not supervised.
