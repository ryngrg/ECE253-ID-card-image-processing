# Next steps

All experiments are done (glare, blur, joint; results in README) and the files are grouped by glare / blur / joint / common.
What is left, in priority order: **1. report, 2. final presentation, 3. repository cleanup.**
No new experiments unless the report needs one.

**Report and presentation are both on 12/5/2026.** The report has at most **5 pages, references included** (working
assumption: the only requirement so far is the ICASSP 2027 template; check again if the course publishes guidelines).
The presentation is 8 min on Zoom (7-10 pm; attend the whole session). The submission will be a zip of the final repo state;
its contents are decided later.
We write everything here; the teammate only presents the blur part, so they need the blur slide and notes early.

| by | done |
|---|---|
| 11/27 | shared figures and tables (done 11/25) |
| 12/1 | full report draft |
| 12/2 | slides and blur speaker notes to the teammate |
| 12/3 | report revised; first rehearsal |
| 12/4 | cleanup; second rehearsal |
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
* Card data in the repo (private on GitHub) stays as it is for now.

---

## Shared figures and tables (done 11/25)
`python src/report/make_tables.py && python src/report/make_figures.py` rebuilds everything from `results/`.
Specimen cards only (F01-F05, B01-B03), plus MIDV-2020 documents.
* Figures, `report/figures/*.pdf` (paper) and `.png` (slides): `pipeline`, `glare_methods` (block diagrams, TikZ), `dataset` (one column);
  `blur_crops`, `glare_crops` (blue = core, orange = halo), `chain_crops`, `fusion`, `halo_changes`, `joint_changes` (full width).
  For slides also `results/figures/` (`core_halo_example.png`, `demo_*.jpg`). Do not use `test_S_G_comparison.jpg` (a real person's card).
* Tables, `report/tables/*.tex` (all fit the page width at footnotesize; `tables.md` to read them): `dataset`, `blur`, `glare_zones`,
  `glare_halo_changes` (same numbers as the `halo_changes` figure), `ablation`, `modifications`, `fusion`, `fusion_changes`, `joint_bg`,
  `joint_sg`, `joint_midv`, `joint_steps`. Not all fit in 5 pages: pick while writing.
* Crops were picked to show both gains and failures; say so in the captions (P = PaddleOCR, T = Tesseract, check mark = all characters right).

---

## 1. Report (ICASSP 2027 template)
5 pages in total, references included (the template's own rule: 4 pages + a 5th with references only).
Abstract 100-150 words, font ≥ 9 pt, no page numbers, not blind, a "relation to prior work" discussion is required.

- [ ] Set up `report/`: `paper.tex` from `Template.tex`, with `spconf.sty`, `IEEEbib.bst`, figures. Compiles with local `latexmk`.
- [ ] `refs.bib`: start from `docs/course/proposal/refs.bib` (36 cited entries; it already has Restormer, NAFNet, MIDV-2020, Tesseract,
      SSIM, Telea, the dichromatic model). Drop what the paper does not cite. Add: DocSHRNet, PaddleOCR, ECC, SIFT, RANSAC, dark channel
      prior (He et al.; `pan2016` is a deblurring paper), Canny / Otsu / GrabCut if rectification is described. Check every new entry at its source.
- [ ] Cover the course requirements (top of `docs/course/proposal/proposal.tex`): own + public data; per artifact ≥ 2 methods, one conventional;
      modified for our data; compared with and without the modification; say whether it helps or not.
- [ ] Draft to this budget:

| section | pages | content |
|---|---|---|
| Abstract, introduction | 0.55 | problem, artifacts, main findings |
| Related work | 0.35 | relation to prior work |
| Data and metric | 0.65 | our 141 photos / 22 cards, MIDV photos and clips, rectification + ECC, split, fixed-box OCR, zones, CIs |
| Methods: blur | 0.45 | Wiener fixed / adaptive, Restormer pretrained / fine-tuned |
| Methods: glare | 0.7 | classical two-zone with statistics-inpainting veil; DocSHRNet + detection head; fusion |
| Results: blur | 0.4 | B_D test (n = 4) |
| Results: glare | 0.65 | S_G test per zone and engine with CIs, MIDV replication, SSIM vs OCR, fusion in one table |
| Results: joint | 0.4 | B_G, MIDV photos, order; deblurring of fused clips and runtime in one sentence each |
| Discussion, conclusion | 0.35 | limitations, lessons, future work |
| References | 0.5 | |

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

- [ ] Build a new deck in `presentation/` (no editable source of the update deck; reuse its content from the PDF and our figures).
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
- [ ] Write speaker notes for the blur slide and send them to the teammate by 12/2.
- [ ] Rehearse together with a timer (7:45-8:15); prepare questions (n is small, why does PaddleOCR get worse, why does Wiener fail).
- [ ] Both members upload the file on 12/5 (the handout's own deadline is midnight 12/07); be ready when called (order unknown).

---

## 3. Clean up the repository
Rule: keep what reproduces a number or figure in the report / slides, or an ablation they mention; remove the rest.
Never delete `data/raw/`, `ground_truth/`, `split.csv`, the frozen parameters or the final checkpoints **from disk**.
Done: files grouped into glare / blur / joint / common (tables recomputed from the OCR cache and the methods' outputs were identical).

### 3a. Remove (candidates)
- [ ] Orphans: `results/glare/whole_card/check_*` (no script uses them).
- [ ] Denoise before veil inversion (no test gain): `--denoise` code in `classical_glare.py`, `cdn_*` and `classical_dn` results,
      `results/glare/denoise_sel/`, its steps in `run_glare_experiments.sh`. One sentence stays in README.
- [ ] Also unused in `classical_glare.py`: CLAHE, descratch.
- [ ] `real-pilot` model: entry in `models.json`, `install_deblur_weights.py`, DEBLUR.md section.
- [ ] `docs/blur_history/`: keep `README.md` (what was tried); drop the 11 raw reports (they stay in the blur repo @ `2893d66`).
- [ ] Whole-card Tesseract runs (`*_tess`) and `analyze_ocr_loss.py` / `ocr_loss_*`, replaced by the fixed boxes: remove unless the report cites them.
- [ ] Chain runs that no table uses (`dlzs_*`, `restormer_pre_dlzs`, `wiener_dl`, `dl_wiener`), after the report is final.
- [ ] `*_boxes.csv`: large and regenerated by `box_ocr.py` (cached OCR); drop.
- [ ] Local only (git-ignored): `out/cdn_*`, `out/classical_dn`, `docshr_mt_last.pt`, intermediate `restormer_ours/` steps, old `logs/`, `.DS_Store`.

### 3b. Keep (they carry report content)
* Decay veil (`params/glare/classical_params_decay.json`; "before our modification"), Telea, veil-only and core-fill-only ablations.
* Wiener fixed / adaptive / old setting and Restormer pretrained ("without our modification"); their tests.
* `ssim_diagnose.py` (SSIM vs OCR), `benchmark_runtime.py`, temporal median and best-frame baselines.

### 3c. Tidy
- [ ] Commit `report/`, `src/report/`, the figures and `ocr_accuracy_by_zone.csv`, if used.
- [ ] No script in `src/` names these files: `midv_by_doctype.csv`, `detection_iou.csv`, `ocr_accuracy_by_zone.csv`, `test_S_G_*`,
      `midv_examples.jpg`. Add a script for each one the report keeps, drop the others.
- [ ] `docs/course/proposal/`: keep `proposal.tex`, `refs.bib` and the style files; drop the template leftovers (`Template.tex` is the
      same file as `proposal.tex`; `format.*`, `Author-PDF-Guide`).
- [ ] Remove `ICASSP2027_Paper_Templates/` once `report/` has what it needs.
- [ ] Delete the stale `logs/`; rewrite DEBLUR.md as plain documentation (install, run, tune) without the history.
- [ ] Merge the three `requirements*.txt` files into one.
- [ ] Shorten README (setup, pipeline, final tables, findings); update DEBLUR.md and CLAUDE.md for everything removed.
- [ ] Check: `python -m pytest -q tests/blur` passes, no dangling references, `bash -n` accepts both `src/*/*.sh`, and the summary / compare
      tables rebuilt from the OCR cache equal the committed ones.
