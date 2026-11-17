# ECE253-ID-card-image-processing
ECE253 ID card image processing: out-of-focus blur and **glare** (specular highlight) removal.
This README covers the glare part.

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
| paired comparison with bootstrap CIs | `src/compare_runs.py` | `results/compare_*.csv` |
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

### Findings
1. **Neither method significantly improves OCR on unseen cards.**
   * Where OCR loses words, they are mostly in the glare: 44% of halo words and 82% of core words are lost,
     versus 11% elsewhere.
   * But the strongest glare is not a smooth veil. It is a textured reflection of the laminate (scratches,
     hologram) in which the strokes are physically gone (see the zoom figure).
   * Both methods recover a few halo words. DL fine-tuned cuts the halo digit loss from 0.50 to 0.44.
   * These changes are the size of PaddleOCR's run-to-run churn: even outside the glare, a touched image gains
     and loses a few words.
2. **SSIM and OCR disagree.**
   * The DL model is far better by SSIM, inside and outside the glare. Zero-shot raises ssim_outside the most
     simply by smoothing the whole card (cf. `ssim_diagnose.py`: light blur raises SSIM from 0.76 to 0.92).
   * It does not read better.
   * Report both, and treat SSIM gains without OCR gains with care.
3. **Fine-tuning matters for the network.**
   * Zero-shot DocSHRNet paints dark text-like smears into clipped cores (hallucination), and its CER is worse
     than untreated (0.135 vs 0.125).
   * After fine-tuning, cores become smooth fills and CER is significantly better than zero-shot (0.109).
   * The gate keeps text outside the glare unchanged.
4. **The classical method is safe but weak.**
   * The veil model is physically right for the halo, and the new estimator removes the harm of the old
     distance-decay veil away from the glare (ssim_outside significantly better than decay on val and test).
   * On the other hand, the decay veil had the higher ssim_glare on val, where its parameters were tuned;
     on test the two are not significantly different.
   * But it acts only near clipped seeds (18 of 94 images untouched) and cannot undo textured specular
     reflection.
   * Inpainting the core does not help OCR. The information there is lost; flat fill at least does not invent
     strokes. Large cores should rather be flagged for recapture, as in the proposal.

Caveats:
* Test has 8 S_G / 9 B_G images from 4 cards. The CIs are wide, and several choices that looked better on
  train+val did not transfer.
* The DL glare masks used for training come from reference differences (> 40 grey levels), so very weak veils
  are not supervised.
