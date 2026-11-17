# Glare removal: next steps

Status (see README "Results"): neither method significantly improves OCR on the 4 test cards.
DL fine-tuned is much better on SSIM. Suggested priority: **1 → 2 → 3 → 6 (multi-frame) if time allows**.

## 1. OCR per region with fixed boxes (main OCR metric)
Glare covers ~10% of the card but holds ~25% of the words, so whole-card CER/F1 dilutes the effect.
Outside the glare, methods should simply match untreated (do-no-harm).
- [ ] Get boxes from the reference's text boxes, or from boxes drawn once around the ground-truth fields.
      Images are aligned, so the same boxes apply to every treated image.
- [ ] Crop the boxes and run **recognition only** (PaddleOCR `TextRecognition`, no detection).
      This removes most of the run-to-run churn, which comes mostly from text detection (boxes splitting and
      merging), and is much faster than full OCR.
- [ ] Tag each box core / halo / outside (overlap logic from `analyze_ocr_loss.py`) and score it against the ground truth.
- [ ] Report per-image accuracy per zone with paired bootstrap CIs (extend `compare_runs.py`), plus a digits-only
      (dates, ID numbers) table.
- [ ] Noise floor: photograph a second S_D image of a few cards. The OCR difference between two clean shots is
      the smallest detectable effect.

## 2. Second OCR engine: Tesseract
Tesseract binarizes, so a raised black level from a veil hurts it more than PaddleOCR: gains may be larger.
Report **both** engines; picking the engine after seeing which one shows bigger gains would be cherry-picking.
- [ ] `python src/evaluate.py --engine tesseract [--treated-dir out/X --name X_tess]` for untreated, classical,
      dl_zeroshot and dl_ft (Tesseract 5.5.3 is installed, ~1 s/image).
- [ ] Also run it on single cropped lines (`--psm 7`, boxes from step 1) to avoid floor effects from the card layout.
- [ ] One results table per engine.

## 3. Public dataset
- [ ] **MIDV-2020** (mock IDs, real phone photos / video, document outlines + field text annotations; check which
      annotations the chosen subset provides). No glare-free pairs, so evaluate OCR vs the ground truth only.
  - Select photos/frames with glare (clipped seeds + a quick manual check).
  - Rectify them with the provided outlines.
  - Run all methods and score field-level OCR accuracy with the step-1 method.
- [ ] **DocHighlight** (~2,200 paired real document highlights, DocSHRNet's training set): PSNR/SSIM on a public
      paired benchmark, and a check that fine-tuning on ID cards did not make the model forget documents.
      The download is only on Baidu/Quark Netdisk: plan time for that or email the authors.
- [ ] Training on external data is allowed: DocHighlight + synthetic pairs (step 6.6). Keep our 4 test cards
      untouched for the final numbers.
- [ ] Report: ours-test, MIDV glare subset (OCR) and DocHighlight-test (SSIM/PSNR).

## 4. Classical refinements (small expected gains)
- [ ] `--denoise`: veil inversion multiplies noise by 1/(1−a) (~7× at a = 0.85). Apply an edge-preserving denoise
      (guided filter / non-local means), weighted by `a`, *before* the inversion. Evaluate on train+val with the
      step-1 metric.
- Not recommended: global denoising / contrast changes (they break do-no-harm; PaddleOCR normalizes contrast
  anyway). CLAHE in the halo already hurt OCR, and the grayscale opening raised SSIM but not OCR.

## 5. Rectification / alignment (good enough; low priority)
118/119 aligned; `ssim_diagnose.py`: best shift (0, 0), exposure explains ~0.009 SSIM.
- [ ] F13_B_G_2: also try the 180°-rotated warp and keep the one with the higher ECC correlation.
- [ ] Give ECC an `inputMask` that excludes clipped / very bright pixels (glare can bias the alignment).
- [ ] For a few images, report the residual displacement with dense optical flow (`cv2.DISOpticalFlow`)
      vs the reference. If it is under a pixel, state that in the report.

## 6. If redoing the project from scratch
1. Fix the metric first: fixed-box OCR per field and zone, with a measured noise floor; SSIM only as a secondary check.
2. **Multi-frame capture + fusion.** Glare moves with the phone or light; the card content doesn't. Align 2–3 frames
   and fuse them robustly (e.g. per pixel, keep the darkest value outside the glare in any frame). This is the only
   approach that recovers clipped cores, and it matches how scanning apps work (the proposal's video stretch goal).
3. **Detect + ask for a recapture** for fields under glare, instead of inventing text. Evaluate as per-field glare
   detection accuracy.
4. Capture protocol:
   - Lock exposure and white balance between the reference and glare shots.
   - Use RAW or exposure brackets to reduce clipping.
   - More cards and more glare shots per card, plus two clean references per card.
5. Card-level 5-fold cross-validation instead of one 4-card test set (DL: ~5 × 1.5 h on M2).
6. **Glare transplant augmentation:** extract each aligned pair's glare layer (veil map `a` + clipping mask, from
   `I = R(1−a) + 255a`) and composite it onto other clean cards (ours + MIDV templates) with random shifts,
   rotations and scales. 94 real layers can produce thousands of realistic training pairs.
7. Train for legibility: add a text-recognition loss (features from a frozen recognizer, as in TASHR
   "text-aware highlight removal"). Keep the detection gate; leave clipped cores flagged rather than
   hallucinating strokes into them.
