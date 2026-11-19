# Glare removal: next steps (in priority order)

## Where we are
The core deliverable is done: a classical and a learning-based method, with and without our modifications, vs
untreated, per glare zone, two OCR engines, on our data, on 100 external MIDV-2020 photos, and with multi-frame fusion
on 100 MIDV-2020 clips. Methods, results and findings are in the README; runtime is in README "Runtime".
The proposal's "approach" section is a guide, not a contract. The items below are ranked by value for the report per
day of work, not by how closely they follow the proposal.

## Decisions (agreed; do not revisit without a reason)
* **No artificial glare**: no synthetic or transplanted glare on MIDV or on our cards.
* **Test data stays untouched**: our 4 test cards are never trained or tuned on; nothing is tuned on the MIDV
  photos (and in step 7 only on the MIDV train types).
* **MIDV is split by document type, never by image**, with one fixed split (no cross-validation); see step 7.
* **DL training recipe is fixed** (1500 iterations, current loss weights, last checkpoint), with no val split, if
  we retrain at all. The val curve is flat after ~1000 iterations (best 1350 vs last 1500: ssim_glare 0.5641 vs 0.5640).
* Any new threshold or parameter (steps 3 and 6) is fixed in advance or chosen on our train + val only, then frozen
  for our test and MIDV.

---

## 1. Write the report (start now; the largest remaining item)
Everything is in the README (~390 lines of results); the report is a short paper in the proposal's format, so most
of the work is choosing what to cut.
- [ ] Confirm the page limit and deadline; agree with the teammate on the shared sections (intro, data, combined
      artifacts, conclusion) and who writes what.
- [ ] Outline with a page budget before writing. Glare essentials: method summary (classical + DL), test S_G box OCR
      per zone for both engines with paired CIs, the MIDV-2020 replication, SSIM vs OCR, fusion in one table
      (keymax core + all fields), runtime in one line. The rest goes to an appendix or the repo link.
- [ ] State the deviations from the proposal and why:
      - JSHDR (Fu et al.) is gone, so DocSHRNet with a new detection head;
      - HSV hysteresis + low-pass veil replaced by the statistics-inpainting veil; Telea core fill is an ablation;
      - the degradation-assessment routing stage (step 3 or dropped);
      - PSNR / LPIPS dropped for the fixed-box OCR metric;
      - the MIDV-2020 photos stand in for the planned hand-held, uncontrolled set (hand-held phone photos with
        ground-truth text).
- [ ] Figures: pipeline, qualitative crops (step 4), `results/figures/midv_clips_fusion.jpg`.

## 2. Blur + glare together: order of the two steps (~1 day; needs the teammate: start the interface now)
Our 47 B_G images (30 / 8 / 9) have both artifacts; this is where the two halves of the project meet, and the
last proposal item not yet done. Expect a null result (B_G halo char accuracy ≤ 0.25 for every method): which order
is less harmful is still a finding.
- [ ] Agree an interface with the teammate: `deblur.py --in-dir DIR --out-dir DIR` on our rectified images,
      keeping the filenames (classical Wiener and the learning-based deblurring).
- [ ] Pipelines on B_G: untreated, deblur only, glare only, deblur → glare, glare → deblur. Pair classical with
      classical and DL with DL. Optionally test the best mixed pair too.
- [ ] Same metrics: fixed-box OCR per zone (both engines), SSIM vs S_D, and `compare_runs.py` on
      test B_G (n = 9; also report train + val as a consistency check).
- [ ] Note in the report: the DL glare model was trained on sharp images only.
- [ ] Start early: it depends on the teammate's code running on our images.

## 3. Recapture flag: "this field is under glare, retake the photo" (~1 day)
From a single image, text in clipped or textured glare is gone and no single-image method brings it back; only
other frames do (README "Multi-frame fusion"), and reading each field from its cleanest frame gets almost all of that
gain. The useful single-image output is then a per-field warning ("retake this field"), which is exactly what triggers
that re-read. This also completes the classical method's "flag large cores as unrecoverable" idea.
- [ ] New script `src/recapture_flag.py`. Unit = one fixed box (field) of one image, from the
      `results/box_{engine}*/untreated_boxes.csv` files (ours + MIDV).
- [ ] Ground truth "needs a retake": the untreated reading of the box is wrong (char accuracy < 0.8, choice fixed
      now) while the reference reading is right. Per engine: PaddleOCR and Tesseract need different retakes.
- [ ] Flag rules to compare (all reference-free, i.e. usable in an app):
      a. box overlaps the clipped core (classical seeds) by ≥ t;
      b. mean classical veil map `a` in the box ≥ t;
      c. mean DL detection-head mask in the box ≥ t (`out/dl_ft/masks/`, `out/midv/dl_ft/masks/`);
      d. a box-level brightness / contrast statistic (e.g. 5th-percentile darkness of the box) ≥ t.
- [ ] Choose each t on our train + val; report precision / recall / F1 of the flag on our test and on all
      100 MIDV photos (external).
- [ ] Also report OCR accuracy on the fields NOT flagged vs the fraction flagged ("retake 12% of the fields →
      accuracy of the rest goes from x to y"). This is the practical result for a scanning app.
- [ ] Optional, same script: the proposal's degradation-assessment stage at image level (variance of the Laplacian
      for blur, fraction of clipped pixels for glare), with its thresholds chosen on train + val. Otherwise the
      report says it was dropped.

## 4. Report figures (~0.5 day)
- [ ] Qualitative figure: zoomed crops of text, hologram and portrait for each method, to look for invented
      characters (zero-shot fills clipped cores with text-like smears) and damage to security features.
      Use our test cards and MIDV examples (`results/figures/midv_examples.jpg`).
- [ ] Optional figure: the halo boxes DL breaks for PaddleOCR (`results/box_paddle*/*_boxes.csv` has every
      reading), to show why (smoothing vs invented texture).

## 5. Noise-floor photos (~30 min, only if the cards are at hand)
- [ ] Photograph a second clean S_D image of 3–4 cards with the same setup, rectify with `rectify_ids.py`, and
      run `box_ocr.py` on them. The difference from the first S_D is the real noise floor, which replaces the
      `ref_jitter` proxy in the README.

## 6. Hybrid: CNN glare detection + classical removal (~1 day; optional, proposal stretch goal 3)
The classical method acts only around clipped seeds: it leaves 69 of 100 MIDV photos and 18 of our 94 images
untouched. The DL detection head finds glare without clipping. Expected gain is small: the head's mask has IoU
0.30 with the glare region on test, and the classical veil inversion is already neutral to harmful for PaddleOCR.
- [ ] `classical_glare.py --mask-dir DIR --mask-thresh t`: use the DL mask (thresholded, dilated) as the region
      where the veil is estimated and inverted. In `estimate_veil`, the mask replaces the seed-based `near`
      zone and the "connected to a seed" hysteresis. `remove_glare` must no longer return early when there are
      no clipped cores. The core fill stays limited to clipped cores.
- [ ] Choose t (and nothing else) on our train + val with the fixed-box metric of both engines, keeping the
      do-no-harm rule for ssim_outside.
- [ ] Evaluate on our test and on all 100 MIDV photos (the DL masks come from the model trained on our data
      only, so MIDV stays external). Compare with classical, DL fine-tuned and untreated.
- [ ] Runtime: `benchmark_runtime.py --methods ...` for the hybrid (it includes the DL forward pass for the mask).

## 7. Optional, only if time remains: train on ours + MIDV (~3–5 days)
Question: does more real ID-card glare close the "fine-tuned ≈ zero-shot on MIDV" gap? It will not change the
headline (engine-dependent) result.

Training target on MIDV = the scan colour-matched to the photo (`prepare_midv.match_tone`). This is a sound
target:
* it is the same printed document, glare-free, aligned to the photo (guarded ECC);
* the colour match removes the device and illumination difference that a raw scan would teach the network to undo;
* restricting the loss to the glare mask leaves the remaining differences outside the glare (sharpness, noise)
  out of training. The network's gate keeps the input there anyway.

Weak points to keep an eye on (they do not argue against it):
* the colour match is smooth (σ = 150 px), so local tone differences remain;
* hologram regions look different in a scanner;
* 31% of MIDV photos had their ECC rejected and are dropped from training.

MIDV split (fixed now, before any results; 2 ID cards + 2 passports in test; write it to `midv_split.csv` and never regenerate it):
| split | types | photos 60–69 | clips 90–99 |
|---|---|---|---|
| **test** | alb_id, esp_id, aze_passport, grc_passport | 40 photos (202 glare-zone boxes) | not used |
| train | est_id, fin_id, svk_id, lva_passport, srb_passport, rus_internalpassport | 60 photos | 60 clips |

rus_internalpassport is in train on purpose: it has ~1 Latin field per document. Expected CI half-width on the 40
test photos is ~0.05 (vs ~0.03 with 100), so a small effect may stay n.s.
- [ ] **Clips:** already aligned for fusion: `out/midv_clips_frames.csv` (one raw -> scan-frame homography per frame;
      `prepare_midv_clips.load_stack` re-warps them). For training, use clips of the 6 train types only, with a frame
      subsample fixed in advance (e.g. every 5th aligned frame), and only clips whose anchor -> scan ECC was used.
- [ ] **`prepare_midv.py --source photo|clips`:** add a `midv_split` column, save the tone-matched scan
      (`data/midv/reference_matched/`) as the DL training target, write `out/pairs_midv_train_clips.csv`
      (not `out/pairs_midv_clips.csv`: that is the fusion key-frame file).
- [ ] **A′ (baseline):** current recipe trained on our train + val S_G (39 images), no val
      (`finetune_dl.py --train-splits train,val --no-val`).
- [ ] **B:** our train + val + MIDV train types (photos + clip frames). On MIDV rows:
      - target = tone-matched scan, with L1 / SSIM only inside the (dilated) glare mask;
      - detection target = the precomputed mask;
      - drop rows with `ecc_used == False`;
      - 50 / 50 crop sampling between ours and MIDV, fixed in advance.
- [ ] **Evaluate:**
      - B vs A′ vs zero-shot vs untreated on the 40 MIDV test photos (box OCR, both engines; `compare_runs.py`
        needs a filter for the test types) and on our test;
      - A′ also on all 100 MIDV photos;
      - on the clips, report the DL baselines on the 4 test types separately (the 6 train types' clips were trained
        on); fusion has nothing learned and stays on all 100.

## Not doing (mention as future work in the report)
* **SHDocs pre-training** (NeurIPS 2024, github.com/JovinLeong/SHDocs; real polarisation-separated pairs on paper
  forms): worth trying only if step 7 runs and fails.
* **Face-recognition evaluation** (stretch goal 2), retuning the classical method with MIDV, cross-validation.
* **Ruled out:** synthetic / transplanted glare. **Dropped:** DocHighlight (the backbone is already pretrained on
  it), DocHR14K (no download link). PSNR and LPIPS (SSIM + the OCR metrics are enough).

## Lessons for the report (if redoing from scratch)
* Fix the metric first: fixed-box OCR per field and zone with a measured noise floor; SSIM only as a secondary check.
* Capture protocol:
  - lock exposure and white balance;
  - use RAW or exposure brackets against clipping;
  - more cards, and two clean references per card;
  - a cross-polarised clean shot (as SHDocs does) gives near-perfect pairs.
* Split by card (and by document type for public sets), make the test set large enough (one 4-card split gave
  CIs of ±0.1), and fix the training recipe instead of selecting on a tiny val split.
* Flag fields under clipped glare for a retake rather than inventing strokes.
* Capture a short burst, not one photo: glare moves with the phone, and a per-field re-read recovers clipped text.
