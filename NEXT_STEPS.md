# Next steps

Steps 1 (deblur on our splits) and 2 (chain deblur and deglare, evaluate) are done; only the report is left.

## Where we are
* **Glare: done.** Classical + DL, with and without our modifications, vs untreated, per zone, two OCR engines, on
  our data, 100 MIDV-2020 photos and 100 MIDV-2020 clips (multi-frame fusion). See README; runtime included.
* **Blur: done.** Wiener (fixed, adaptive) and Restormer (pretrained, fine-tuned) tuned / trained on our train + val, frozen,
  evaluated on B_D test. See README "Deblurring" and DEBLUR.md.
* **Joint: done.** Both orders on our B_G / S_G (all splits), MIDV photos, and deblurring of the fused clip images.
  See README "Joint deblur + deglare". Not run on purpose: single-image chains on the clips; deblurring single frames
  before fusion (4% of the frames are dropped for blur).

## Decisions (agreed; do not revisit without a reason)
* `split.csv` is the only split for both artifacts (14 / 4 / 4 cards). Test cards (B03, F03, F04, F11) are never
  trained or tuned on, in any condition (S_D, S_G, B_D, B_G).
* Nothing is trained or tuned on MIDV; it stays an external test set.
* No artificial glare. The glare methods stay frozen (`classical_params.json`, `docshr_mt_best.pt`).
* Parameters are chosen on our train + val only, then frozen before anything runs on test or MIDV.
* Main metric: fixed-box OCR (`box_ocr.py`, both engines) with paired CIs (`compare_runs.py`); SSIM is secondary.
* The proposal's approach section is a guide, not a contract.

---

## 1 and 2: open points (optional, not started)
- [ ] With the teammate: agree what counts as "our modification" per deblur method (proposal: Wiener fixed vs adaptive;
      Restormer pretrained vs fine-tuned), and share `restormer_ours.pt` (hash in DEBLUR.md).
- [ ] Optional: more mixed classical / learned pairs on our images and MIDV.

## Write the report
- [ ] Confirm the page limit and deadline; agree with the teammate who writes what (blur / glare / shared: intro, data,
      joint results, conclusion).
- [ ] Outline with a page budget before writing.
      - Blur: methods, B_D results.
      - Glare: methods (classical + DL), test S_G box OCR per zone for both engines with paired CIs, MIDV replication,
        SSIM vs OCR, fusion in one table (keymax core + all fields).
      - Joint: B_G, MIDV photos, clips; effect of the order.
      - Runtime in one line. The rest goes to an appendix or the repo link.
- [ ] State the deviations from the proposal and why:
      - JSHDR (Fu et al.) is gone, so DocSHRNet with a new detection head;
      - HSV hysteresis + low-pass veil replaced by the statistics-inpainting veil; Telea core fill is an ablation;
      - which Wiener is the main method (fixed or the proposal's automatic radius / noise estimation), after step 1;
      - the degradation-assessment routing stage was dropped;
      - PSNR / LPIPS replaced by the fixed-box OCR metric; face recognition not evaluated;
      - MIDV-2020 photos stand in for the planned hand-held, uncontrolled set.
- [ ] Figures: pipeline; qualitative zoomed crops per method (text, hologram, portrait; look for invented characters);
      core / halo example; one B_G chain example; fusion example. Start from `results/figures/`
      (`core_halo_example.png`, `demo_*.jpg`, `midv_examples.jpg`, `midv_clips_fusion.jpg`) and `src/make_demo_figures.py`.
- [ ] Limitations: small test split (4 cards; 8 S_G / 9 B_G / 4 B_D), only a proxy for the noise floor, DL glare
      trained on 31 sharp images.
- [ ] Lessons: fix the metric first; lock exposure / white balance and take two clean references per card; more cards and
      a larger test set; capture a short burst instead of one photo; flag clipped fields for a retake instead of
      inventing strokes.
- [ ] Future work (not doing): per-field recapture flag; CNN detection + classical removal hybrid; training on MIDV;
      a second S_D photo per card for a real noise floor; SHDocs pre-training; face-recognition evaluation.
