# Next steps

All experiments are done (glare, blur, joint; results in README) and the files are grouped by glare / blur / joint / common.
Report and slides have full drafts (11/25) and the repository is cleaned up (11/26, section 3). What is left: reviews,
rehearsals and submission (sections 1 and 2), plus two small cleanup decisions (3c). No new experiments unless the report needs one.

**Report and presentation are both on 12/5/2026.** The report has at most **5 pages, references included** (working
assumption: the only requirement so far is the ICASSP 2027 template; check again if the course publishes guidelines).
The presentation is 8 min on Zoom (7-10 pm; attend the whole session). The submission will be a zip of the final repo state;
its contents are decided later.
We write everything here; the teammate only presents the blur part, so they need the blur slide and notes early.

| by | done |
|---|---|
| 11/27 | shared figures and tables (done 11/25) |
| 12/1 | full report draft (first draft 11/25) |
| 12/2 | slides and blur speaker notes to the teammate (first draft of both 11/25) |
| 12/3 | report revised; first rehearsal |
| 12/4 | cleanup (done 11/26); second rehearsal |
| 12/5 | submit report, zip and slides (**each member uploads the slide file**) before 7 pm; present |

## Decisions (agreed; do not revisit without a reason)
* `split.csv` is the only split for both artifacts (14 / 4 / 4 cards). Test cards (B03, F03, F04, F11) are never
  trained or tuned on, in any condition (S_D, S_G, B_D, B_G).
* Nothing is trained or tuned on MIDV; it stays an external test set.
* No artificial glare. All methods are frozen (`params/glare/classical_params.json`, `docshr_mt_best.pt`, `params/blur/*.json`,
  `restormer_ours.pt`).
* Main metric: fixed-box OCR (`box_ocr.py`, both engines) with paired CIs (`compare_runs.py`); SSIM is secondary.
* The old blur pilot numbers (`docs/blur_history/`, update deck) are never reported: its model saw test, val and MIDV data.
* The proposal's approach section is a guide, not a contract.
* With / without our modification, blur: fixed Wiener -> adaptive Wiener (the proposal's version); Restormer pretrained -> fine-tuned.
* Real cards stay private (11/26, section 3d): no photo, ground truth or text of the 14 real cards is in the repository,
  its history or the submission zip; only the 8 specimen cards are shown or shared. Scores of all 22 cards stay.

---

## Shared figures and tables (done 11/25)
`python src/report/make_tables.py && python src/report/make_figures.py` rebuilds everything from `results/`.
Specimen cards only (F01-F05, B01-B03), plus MIDV-2020 documents.
* Figures, `report/figures/*.pdf` (paper) and `.png` (slides): `pipeline`, `glare_methods` (block diagrams, TikZ), `dataset` (one column);
  `blur_crops`, `glare_crops` (blue = core, orange = halo), `chain_crops`, `fusion`, `halo_changes`, `joint_changes` (full width).
  Whole-card panels: `results/figures/demo_*.jpg` (`make_demo_figures.py`, specimen card + MIDV) and `midv_clips_fusion.jpg` (`fuse_frames.py --figure`).
* Paper versions: `blur_crops_small`, `glare_crops_small`, `glare_small`, `fusion_small`, `joint`. Tables, `report/tables/*.tex` (all fit the page width at footnotesize; `tables.md` to read them): `dataset`, `blur`, `glare_zones`,
  `glare_halo_changes` (same numbers as the `halo_changes` figure), `ablation`, `modifications`, `fusion`, `fusion_changes`, `joint_bg`,
  `joint_sg`, `joint_midv`, `joint_steps`. Not all fit in 5 pages: pick while writing.
* Crops were picked to show both gains and failures; say so in the captions (P = PaddleOCR, T = Tesseract, check mark = all characters right).

---

## 1. Report (ICASSP 2027 template)
5 pages in total, references included (the template's own rule: 4 pages + a 5th with references only).
Abstract 100-150 words, font ≥ 9 pt, no page numbers, not blind, a "relation to prior work" discussion is required.

**First draft done (11/25): `report/paper.tex` -> `paper.pdf`, 5 pages including 26 references, 9 pt, abstract 139 words.**
Build: `cd report && latexmk -pdf paper.tex`. There is no slack: new text needs a cut.
- [ ] Both authors read it; check tone and the deviations paragraph. (Capture: hand-held, mostly iPhone 17, some 12 Pro, focus locked on another object.)
- [ ] Left out for space; decide if anything must come back: `glare_methods` and `chain_crops` figures, `ablation`, `joint_sg`, `dataset` tables.
- [ ] In-figure text is about 7 pt (the template asks for 9 pt everywhere); tables and captions are 9 pt.
- [ ] `refs.bib`: page numbers of `aliev2021` missing; `cho2012`, `lee2021`, ... have no pages (as in the proposal).
- Course requirements covered: own + public data; two methods per artifact, one classical; modified and compared with / without (Table 3).

The items below are the checklist the draft was written to.
- [ ] Main messages:
      - blur: fine-tuned Restormer helps, Wiener does not transfer; Restormer can change digits (03/07/2022 -> 2922), a risk for IDs;
      - glare: the effect of single-image removal depends on the engine (helps Tesseract, slightly hurts PaddleOCR), replicated on MIDV;
      - clipped text comes back only with several frames; reading each field from its best frame gets almost all of that gain;
      - joint: deblurring does most of the work, deglare after it adds a little for Tesseract only, the order does not matter;
      - SSIM and OCR disagree;
      - modifications (`modifications` table): fine-tuning clearly helps both networks on our test cards (n = 4 / 7), but DocSHRNet's gain
        over zero-shot does not carry over to MIDV; the statistics veil helps Tesseract only, adaptive Wiener raises SSIM, not OCR.
- [ ] State the deviations from the proposal and why:
      - JSHDR (Fu et al.) is gone, so DocSHRNet with a new detection head;
      - HSV hysteresis + low-pass veil replaced by the statistics-inpainting veil; Telea core fill is an ablation;
      - the degradation-assessment routing stage was dropped;
      - PSNR / LPIPS replaced by the fixed-box OCR metric; face recognition not evaluated;
      - MIDV-2020 photos stand in for the planned hand-held, uncontrolled set; DPDD / RealBlur / SHIQ / SmartDoc-QA not used;
      - the data are not only mock cards: 14 of the 22 card sides are real cards;
      - Restormer fine-tuned on real pairs only (no synthetic blur).
- [ ] Limitations: bootstrap CIs with n = 4 (B_D) are rough; small test split (4 cards; 8 S_G / 9 B_G / 4 B_D), only a proxy for the noise floor, DL glare
      trained on 31 sharp images, Restormer on 16 blurred ones.
- [ ] Lessons: fix the metric first; lock exposure / white balance and take two clean references per card; more cards and
      a larger test set; capture a short burst instead of one photo; flag clipped fields for a retake instead of
      inventing strokes.
- [ ] Future work (not doing): per-field recapture flag; CNN detection + classical removal hybrid; training on MIDV;
      a second S_D photo per card for a real noise floor; SHDocs pre-training; face-recognition evaluation.
- [ ] Final check: every number against `results/`, ≤ 5 pages, fonts embedded.

---

## 2. Final presentation (`docs/course/handout6_final_presentation.pdf`)
8 min ± 15 s, both members speak (teammate: blur slide; Aaryan: the rest), pictures and block diagrams.
Seven slides; everything else goes to backup slides.

| slide | time | content |
|---|---|---|
| Title | 0:15 | title, members |
| Objective and artifacts | 1:00 | goal, blur and glare (core / halo picture), metrics: fixed-box OCR with two engines, SSIM |
| Dataset | 1:00 | ours (22 cards, 141 photos, table, examples), MIDV photos + clips, split |
| Blur | 1:30 | block diagram Wiener / Restormer, crops, B_D table |
| Glare | 1:45 | block diagram classical / DL / fusion, crops, compact zone table |
| Combined | 1:30 | both orders on B_G. Best: Restormer ft → DL ft for Tesseract; for PaddleOCR Restormer alone does as well |
| Conclusion | 1:00 | findings, limitations, advice: capture a short burst |

**First draft done (11/25):** https://claude.ai/artifact/8oJ6Df6vgTKXC5wrxfsDAT (private; share it from its Share menu). 7 slides + 5 backup slides,
speaker notes on every slide (about 980 words: 7.0 min at 140 words/min, 7.9 min at 125). Pictures: `presentation/figures/`
(`make_figures.py --only slides`; `slide_dataset.png` is a copy of `report/figures/dataset.png`).
- [ ] Review the slides; share the deck with the teammate.
- [ ] Export the final version into `presentation/` (the PPTX of 11/25 is there; add a PDF).
- [ ] Reuse: core / halo slide, dataset tables, glare OCR table (its numbers still match the README), appendix figures.
- [ ] Fix what is outdated in `docs/course/project_update_presentation.pdf`:
      - slides 3, 12 and appendix E use the old blur pilot (invalid, see Decisions): redo with the README B_D numbers and `restormer_ours`;
      - slide 2: one shared metric for both artifacts, not CER for blur and box accuracy for glare;
      - slide 5: the removal head is DocSHRNet's own pretrained output; only the detection head is new, and the backbone is fine-tuned too;
      - slide 10: "multi-frame fusion is unbeaten" → best frame per field gets almost all of the gain; label n and that "MIDV clips" = keymid;
      - slide 11: the SSIM table does not show overfitting (zero-shot wins by smoothing the whole card); the evidence is the
        detection IoU, 0.59 on train vs 0.30 on test;
      - slides 5-7 and 13 (methods in full text, parameters, next steps): shorten, move to backup, or drop;
      - spelling: Wiener, Restormer.
- [ ] Send the blur slide and its notes (written) to the teammate by 12/2.
- [ ] Rehearse together with a timer (7:45-8:15); prepare questions (n is small, why does PaddleOCR get worse, why does Wiener fail).
- [ ] Both members upload the file on 12/5 (the handout's own deadline is midnight 12/07); be ready when called (order unknown).

---

## 3. Clean up the repository (done 11/26)
Rule: keep what reproduces a number or figure in the report / slides, or an ablation they mention; remove the rest.
Never delete `data/raw/`, `ground_truth/`, `split.csv`, the frozen parameters or the final checkpoints **from disk**.

### 3a. Done
* Checked first: `make_tables.py`, `summarise_chain.py` and `evaluate.py --combine` rebuilt the committed tables
  byte for byte. After the cleanup `report/tables/` is unchanged, the summary tables only lost the rows of removed
  runs, and the classical method reproduces all five variants on all 94 glare images bit for bit.
* Removed (code and results): the denoise variants (`--denoise`, `cdn_*`, `classical_dn`, `denoise_sel/`), CLAHE and
  descratch; whole-card Tesseract runs (`*_tess`) and `analyze_ocr_loss.py` / `ocr_loss_*`; the orphan `check_*`; the
  chains no table uses (`dlzs`, `dlzs_restormer_pre`, `restormer_pre_dlzs`, `wiener_dl`, `dl_wiener`); the `real-pilot`
  model entry; easyocr and the PaddleOCR 2.x branch in `evaluate.py` (default engine now PaddleOCR).
  README keeps one sentence on what was tried.
* Files without a script: `detection_iou.csv` (cited in the paper) now comes from `src/glare/detection_iou.py`
  (same numbers); `midv_by_doctype.csv`, `ocr_accuracy_by_zone.csv` (= the `glare_zones` table), `midv_examples.jpg`,
  `core_halo_example.png`, `test_S_G_*` (a real person's card) are gone.
* Kept, unlike the earlier plan: `*_boxes.csv`, because `make_figures.py` reads the per-box readings for the crop figures.
* Docs: one README (shorter: setup, data, evaluation, methods incl. blur, reproduce, final tables from `tables.md`,
  findings); `DEBLUR.md` -> `src/blur/README.md` (block usage only, no history); `docs/blur_history/` = its README
  (the raw reports stay in the blur repo @ `2893d66`); one `requirements.txt` (pinned to the tested versions);
  CLAUDE.md updated. MIDV clips download moved into `prepare_midv_clips.py`.
* Templates: `ICASSP2027_Paper_Templates/` -> `docs/course/icassp2027_template/` (git-ignored; `report/` has the style
  files and builds without it); proposal leftovers (`Template.tex` = `proposal.tex`, `format.*`, `Author-PDF-Guide`) removed.
* `report/` and `presentation/` stay at the top level: they are the final deliverables and are written by
  `src/report/`; `docs/course/` holds the course material and the earlier deliverables (proposal, update slides).
* Checks: `pytest tests/blur` (20 passed), `bash -n` on both scripts, `py_compile` + `--help` of the changed scripts,
  the paper builds (5 pages).

### 3b. Kept (they carry report content)
* Decay veil (`params/glare/classical_params_decay.json`; "before our modification"), Telea, veil-only and core-fill-only ablations.
* Wiener fixed / adaptive / old setting and Restormer pretrained ("without our modification"); their tests.
* `ssim_diagnose.py` (SSIM vs OCR), `detection_iou.py`, `benchmark_runtime.py`, temporal median and best-frame baselines.
* `make_demo_figures.py` + `results/figures/demo_*.jpg` (not in the paper / slides; whole-card views for readers).

### 3d. Real cards redacted (11/26)
* Only the 8 specimen cards' text is tracked: real cards' `ground_truth/` files are git-ignored (kept on disk), and
  `box_layout.csv` / `*_boxes.csv` keep every row and number with the real cards' text blank. Full versions: `private/`
  (git-ignored, local). `src/common/privacy.py` (`save_csv` / `read_csv`) keeps it that way for every new run;
  `tests/common/test_privacy.py` checks it. Verified: box OCR from the cache reproduces the committed scores, tables
  and figures are unchanged.
* The old history had real-card text and the image of a real person's card, so every commit was rewritten (real-card
  ground truth and that image removed, real-card text blanked; same files at the end) and the GitHub repo was deleted
  and recreated with the rewritten history (this removed the old PR ref too). Local backups only:
  `../ECE253-full-history-backup.bundle` (old history, unredacted) and `../ECE253-redacted-history.bundle` (the
  rewritten commits). The teammate must re-clone.

### 3c. Left
- [x] Local only (git-ignored), deleted 11/26: `out/cdn_*`, `out/classical_dn`, `out/chain/{dlzs_restormer_pre,restormer_pre_dlzs,wiener_dl,dl_wiener}`
      (outputs of removed runs, ~1 GB), `logs/` (stale), `.DS_Store` files (except under `data/`).
- [x] Non-final checkpoints deleted 11/26 (local, ~530 MB): `checkpoints/glare/docshr_mt_last.pt` and
      `checkpoints/blur/restormer_ours/step_*.pt` (step 1000 was byte-identical to the kept `restormer_ours.pt`;
      redoing `tune_deblur.py restormer` now needs `finetune_deblur.py` first).
- [x] Zip contents decided (11/26): `python src/report/make_submission_zip.py` -> `../ECE253-ID-card-image-processing.zip`
      (~7.3 GB): the working tree with our data, outputs and fine-tuned weights, without the real cards, `private/`,
      public downloads (MIDV-2020, DocSHRNet, pretrained Restormer, ICASSP kit), git metadata and caches.
      Rebuild it after the last commit before submitting.
