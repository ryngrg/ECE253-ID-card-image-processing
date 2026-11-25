# Next steps

Three things are left, in this order: (1) put the deblur methods on our splits, (2) chain deblur and deglare and
evaluate, (3) write the report. Report writing can start in parallel (the glare half is finished).

## Where we are
* **Glare: done.** Classical + DL, with and without our modifications, vs untreated, per zone, two OCR engines, on
  our data, 100 MIDV-2020 photos and 100 MIDV-2020 clips (multi-frame fusion). See README; runtime included.
* **Blur: all code is in this repo, not yet tuned, trained or evaluated on our split.** The training / tuning code and the
  adaptive Wiener were ported from the teammate's repo (now only an archive; see `docs/blur_history/README.md`).
  * The old fine-tuned Restormer is not usable: it was trained on F03 (our test), F01 (our val) and MIDV frames
    (one document is in our external clips).
  * `deblur_wiener_params.json` still has the old radius 3 / K 0.3, chosen at native resolution (about 1 px in ours).
* **Joint evaluation: not started.**

## Decisions (agreed; do not revisit without a reason)
* `split.csv` is the only split for both artifacts (14 / 4 / 4 cards). Test cards (B03, F03, F04, F11) are never
  trained or tuned on, in any condition (S_D, S_G, B_D, B_G).
* Nothing is trained or tuned on MIDV; it stays an external test set.
* No artificial glare. The glare methods stay frozen (`classical_params.json`, `docshr_mt_best.pt`).
* Parameters are chosen on our train + val only, then frozen before anything runs on test or MIDV.
* Main metric: fixed-box OCR (`box_ocr.py`, both engines) with paired CIs (`compare_runs.py`); SSIM is secondary.
* The proposal's approach section is a guide, not a contract.

---

## 1. Deblur on our dataset splits (teammate + us)
Goal: one frozen Wiener setting and one fine-tuned Restormer checkpoint that have never seen a test card.

**Done:** `einops` + official weights installed, tests pass, `out/pairs_blur.csv` (16 / 4 / 4), ported scripts smoke-tested.
All 8 val / test B_D images have a reference of the same size and aligned status; F11_B_D_0's low ECC (0.58) is a lighting
gradient on its reference, not misalignment (checked by eye).

**Agree with the teammate**
- [ ] What counts as "our modification" for each deblur method (the report needs with / without). Proposal:
      Wiener fixed vs adaptive (the proposal's automatic radius / noise); Restormer pretrained vs fine-tuned.
- [ ] Look through the train overlays (`out/overlays/blur_*`) before training.

**Tune / train (train + val only; commands in DEBLUR.md; selection rule fixed in `tune_deblur.py`)**
- [ ] `tune_deblur.py wiener --write` (~1 h for 49 candidates x 20 images, both engines).
- [ ] `tune_deblur.py wiener_adaptive --write`.
- [ ] `finetune_deblur.py` (2000 steps; the pilot took 29 min on the teammate's machine), then `tune_deblur.py restormer --write`.
      B_G is not used for training (glare would leak into the deblurring target).
- [ ] Record the checkpoint hash in `models.json` / DEBLUR.md and share the file with the teammate.

**Evaluate deblur alone (with the parameters frozen)**
- [ ] On B_D: untreated, Wiener (old setting / retuned / adaptive), Restormer (pretrained / fine-tuned), reference ceiling.
      Fixed-box OCR with both engines, SSIM vs S_D, paired CIs. Test has n = 4 images, so also report train + val.
- [ ] Run `box_ocr.py` with `--pairs out/pairs_blur.csv --layout results/box_layout.csv --tag blur`: without `--layout` / `--tag` it
      overwrites the shared layout and the glare results.
- [ ] Use `char_acc_all` on B_D: there is no glare, so the zones carry no meaning there (blurred strokes can still trip
      the > 40 grey-level rule).
- [ ] Runtime: add Wiener and Restormer to `benchmark_runtime.py`.

## 2. Chain deblur and deglare, evaluate jointly
**Interface (small code change)**
- [ ] Deglare → deblur already works: `deblur.py --in-dir out/classical --out-dir ...`.
- [ ] Deblur → deglare does not: `classical_glare.py` and `finetune_dl.py --infer` read the `path` column of the pairs
      file. Add `--in-dir DIR` to both (read `DIR/<filename>` instead), so the pairs file, references and zones
      stay those of the untreated image.
- [ ] `deblur.py` processes a whole directory and refuses an existing output directory: the run script has to handle reruns.
- [ ] Add a `chain` step to `run_glare_experiments.sh`.

**Runs** (the same list on every dataset; outputs in `out/chain/<run>/`, `out/midv/chain/...`, `out/midv_clips/chain/...`)
| | classical | learning-based |
|---|---|---|
| deblur only | Wiener | Restormer ft |
| deglare only (exists) | classical | DL ft |
| deblur → deglare | Wiener → classical | Restormer ft → DL ft |
| deglare → deblur | classical → Wiener | DL ft → Restormer ft |

Plus untreated and reference. Optional, only if it is cheap: pretrained Restormer + zero-shot DocSHRNet as the
"without modifications" chain, and one mixed pair. Both orders are always reported; no order is picked on test.

**Our B_G (30 / 8 / 9): the main joint result**
- [ ] Box OCR per zone and over all fields (both engines), SSIM vs S_D, `compare_runs.py` on test (n = 9), with
      train + val as a consistency check (DL train-card numbers are optimistic).
- [ ] Paired comparisons: each chain vs untreated, vs deblur only, vs deglare only, and order vs order.
- [ ] S_G as a do-no-harm check: the same chains on sharp images (does the deblur step damage them?).
- [ ] Things to look at: Wiener ringing around clipped cores, whether deblurring changes the clipped seeds the classical
      method starts from, and that DL glare was trained on sharp images only. Until now no glare method helped B_G
      (halo char accuracy ≤ 0.25), so blur is the bottleneck.

**MIDV-2020 photos (100): external**
- [ ] Same runs with the existing MIDV flags (`--pairs out/pairs_midv.csv --splits midv --conditions S_G`; box OCR with
      `--layout results/box_layout_midv.csv --tag midv --fold-accents --pad 2`; see CLAUDE.md).
- [ ] These photos are labelled for glare, not blur. They are softer than the scans (median variance of Laplacian
      114 vs 558), but that may be resampling rather than defocus. So the question here is whether adding a deblur
      step helps or harms, not how well defocus is removed.
- [ ] Wiener's radius was set in our rectified pixels; the resolution matches (~11.8 px/mm), the blur size may not.

**MIDV-2020 clips (if possible)**
- [ ] Feasible: the same chains on the key frames (`data/midv_clips/rectified`, keymid + keymax), and
      **fused → deblur** (`out/midv_clips/fused`) and best frame per field → deblur. Compare with fused alone.
- [ ] Plan the time: Restormer takes ~20 s per card on MPS (more for passports), i.e. at least ~35 min per run on the 100 photos and
      ~70 min on the 200 key frames, with two Restormer runs per dataset.
- [ ] Skip deblur → fuse for Restormer (4,641 frames); Wiener only if there is time.
- [ ] Expect little: blurred frames are already dropped before fusion, and fused is close to the ceiling
      for PaddleOCR (core 0.99); Tesseract has a bit more room.

**Write-up of this step**
- [ ] README: one section for the deblur results and one for the joint results; update CLAUDE.md status.

## 3. Write the report
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
