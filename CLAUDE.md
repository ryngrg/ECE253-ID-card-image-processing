# CLAUDE.md

ECE253 project (UCSD): improve ID-card photos for OCR. Two artifacts, one per teammate:
out-of-focus **blur** (teammate) and **glare** = specular highlights (this repo's glare code).
For each artifact: one classical method + one learning-based method (fine-tuned pretrained net),
compared with and without our modifications and against the untreated image. The plan is in `docs/course/proposal/proposal.tex`;
current methods and results are in `README.md`; planned work is in `NEXT_STEPS.md`.

## Data (never delete anything in `data/raw/`)
- 141 PNGs of 22 cards (B01-B08, F01-F14). Name: `{card}_{B|S}_{G|D}_{num}.png`
  (B/S = blur/sharp, G/D = glare/diffuse).
- Reference for any glare image = the same card's `S_D_0`.
- Rectified, reference-aligned copies are in `data/rectified/*_rectified.png` (1012x638).
  `data/rectified/alignment_log.csv` holds the per-image status.
- `ground_truth/{card}.txt`: hand-typed card text (used for f1_gt / recall_gt).
- **Privacy:** 14 cards are real IDs (B04-B08, F06-F14); only the specimens B01-B03, F01-F05 may appear in anything shared
  (figures, slides, tracked text). The real cards' ground truth is git-ignored, and tables with card text are written only
  with `common.privacy.save_csv` (blank text for real cards in the tracked file, full copy in git-ignored `private/`) and
  read with `privacy.read_csv`. Never commit OCR text dumps (`results/**/ocr_text/` is ignored). Scores stay for all 22 cards.
- `split.csv`: card-level 14/4/4 train/val/test split. Never regenerate without `--resplit`.
  **Test is for final numbers only**: tune and select on train/val.
- `out/pairs.csv`: one row per glare image (S_G / B_G) with ref_path and split. Always filter `usable == True` (94 of 95).
- `out/pairs_blur.csv`: one row per B_D image (16 / 4 / 4), same columns and split; deblurring trains / tunes on its train + val.
- External test set (never tune on it): MIDV-2020 "highlight present" photos (types x 60-69), raw in
  `data/external/midv2020/` (public FTP mirror), prepared by `prepare_midv.py` into `data/midv/{rectified,reference,glare_masks}/`,
  `out/pairs_midv.csv` (split = midv, condition = S_G) and `results/layouts/box_layout_midv.csv`. Reference = the scan.

## Environment
- venv: `card_scanning_env/bin/python` (Python 3.10, torch with MPS on an M2 Mac, no CUDA). Dependencies: one `requirements.txt`
  (+ the tesseract program); keep the PaddleOCR pins, the readings depend on its models.
- Run every script from the repo root: `python src/<group>/<script>.py`. Every script puts `src/` on `sys.path`; imports name the group
  (`from common.evaluate import ...`, `from glare.classical_glare import ...`). New shared code goes in `common/`.
- Groups `common` / `glare` / `blur` / `joint` are used in `src/`, `results/`; `glare` / `blur` also in `params/`, `checkpoints/`; `tests/` has `blur/` and `common/`.
  Do not move `out/`, `data/` or `results/ocr_cache/`: the OCR cache is keyed by absolute image path + mtime.
- `third_party/DocSHRNet` (code) and `third_party/weights/docshrnet.pth` are git-ignored; README "Setup" has every download
  (software, pretrained networks, MIDV-2020 photos and clips).
  Also git-ignored: `data/`, `out/` (except `out/*.csv`), weights in `checkpoints/`, `logs/`, `results/ocr_cache/`.
- PaddleOCR (the engine we report) takes ~6.5 s/image on CPU, so a full evaluation of 94 images takes ~10 min.
  Results are cached in `results/ocr_cache/<engine>_<lang>/`, keyed by path + mtime: rewriting an output re-OCRs it.
  Run long jobs in the background.

## Scripts (`src/`)
Groups: `common/` rectify_ids, prepare_data, prepare_midv, evaluate, box_ocr, compare_runs, privacy; `glare/` classical_glare, finetune_dl,
prepare_midv_clips, fuse_frames, detection_iou, ssim_diagnose, run_glare_experiments.sh; `blur/` deblur,
deblur_blocks/, tune_deblur, finetune_deblur, install_deblur_weights; `joint/` run_chain_experiments.sh, summarise_chain, benchmark_runtime.
- `rectify_ids.py`: card detection, warp to 1012x638, ECC homography alignment to the S_D reference.
- `prepare_data.py`: builds `out/pairs.csv` and `split.csv`, plus overlays in `out/overlays/`.
- `classical_glare.py`: two-zone classical method.
  - Clipped seeds, then a veil estimated by "statistics inpainting" (`veil_mode=stat`), white-level cap,
    background floor, and veil inversion on RGB.
  - Flat core fill (`core_fill=flat`; `telea` is the ablation).
  - `--tune` runs an SSIM grid with the do-no-harm rule.
  - Params: `params/glare/classical_params.json` (final) and `params/glare/classical_params_decay.json` (old distance-decay model, ablation).
  - Flags: `--no-veil`, `--no-inpaint`, `--core-fill`, `--debug`. (Denoise, CLAHE and descratch were tried, dropped and removed.)
- `finetune_dl.py`: DocSHRNet backbone with a new detection head; output = m·removed + (1-m)·input.
  - Trains on train S_G; picks the best checkpoint on val S_G.
  - `--infer --checkpoint checkpoints/glare/docshr_mt_best.pt` or `--infer --zero-shot`.
- `evaluate.py`: per-image metrics:
  - SSIM: whole image / glare region / outside the glare.
  - OCR: CER and word_f1 vs the reference's OCR, plus f1_gt / recall_gt vs the ground truth.
  - Usage: `--treated-dir out/X --name X` (PaddleOCR is the default engine); `--combine` builds `results/glare/whole_card/all_methods.csv`.
  - Treated images must keep the rectified filenames.
- `box_ocr.py`: **main OCR metric**. Fixed reference line boxes, recognition only, char accuracy per zone vs the GT.
  - `--engine paddle|tesseract`; writes `{--results-dir}/box_{engine}[_tag]/` (default `results/glare`; `results/blur`, `results/joint` for those runs); `--only --extra dir:name` adds runs.
  - Crops use no padding (neighbouring lines leak in) and batch_size=1 (batching pads crops and is 4x slower).
  - `reference` = ceiling, `ref_jitter` = noise-floor proxy.
- `compare_runs.py`: paired per-image differences with 95% bootstrap CIs (`results/glare/whole_card/compare_*.csv`).
  - With `--results-dir results/glare/box_paddle` it compares box metrics; `--metrics` picks a subset.
- `detection_iou.py`: IoU of each method's glare region with the reference-derived one -> `results/glare/detection_iou.csv`
  (the report cites the DL head: 0.59 on train, 0.30 on test).
- `ssim_diagnose.py`: why SSIM stays low outside the glare (fine detail, not misalignment).
- `prepare_midv.py`: MIDV-2020 external set: quad warp + guarded ECC to the scan, template field boxes,
  tone-matched scan for glare masks. Methods run with `--pairs out/pairs_midv.csv --splits midv --conditions S_G --out-dir out/midv/<m>`;
  box OCR with `--pairs out/pairs_midv.csv --layout results/layouts/box_layout_midv.csv --splits midv --tag midv --out-root out/midv --fold-accents --pad 2`.
- `prepare_midv_clips.py`: MIDV-2020 highlight clips 90-99 (frames in `data/external/midv2020/clips/`).
  - Aligns every annotated frame to one anchor per clip (SIFT + normalised ECC, residual guard).
  - Aligned frames are NOT saved: `out/midv_clips_frames.csv` stores one raw->anchor homography + gain per frame, and `load_stack()` re-warps them.
  - Key frames = splits `keymid` / `keymax` of `out/pairs_midv_clips.csv` (images `{type}_{nn}_{mid|max}.png`); layout `results/layouts/box_layout_midv_clips.csv`.
- `fuse_frames.py`: fusion (replace only the key frame's glare pixels) + `median` ablation + `best_frame` / `best_field` baselines
  -> `out/midv_clips/<method>/`, diagnostics `results/glare/midv_clips_fusion.csv`. Box OCR with `--tag midv_clips` (see `run_glare_experiments.sh` step `clips`).
- Deblurring (`src/blur/README.md`): `deblur.py --method wiener|wiener_adaptive|restormer` (directory -> new directory, ~20 s/image for Restormer on MPS),
  `finetune_deblur.py` (Restormer on train B_D), `tune_deblur.py wiener|wiener_adaptive|restormer --write` (selection on train / val).
  The teammate's earlier blur pilot saw F03 (test), F01 (val) and MIDV: never report it (summary in `docs/blur_history/README.md`).
  Tests: `python -m pytest -q tests` (blur blocks + `tests/common/test_privacy.py`; needs `pytest<9`).
- `benchmark_runtime.py`: s/image of each method (processing call only, MPS + CPU) -> `results/runtime/runtime_{name}.csv`.
- `run_chain_experiments.sh` (steps `blur ours midv clips` + `_eval`) and `summarise_chain.py`: deblur-only (-> `results/blur/`) and deblur + deglare (-> `results/joint/`) experiments.
- `src/report/`: `make_tables.py` (every table + `paired_changes.csv` -> `report/tables/`, never type numbers by hand), then `make_figures.py`
  (-> `report/figures/` pdf + png; `--only NAME`); `make_demo_figures.py` (whole-card panels -> `results/figures/`).
  Paper: `report/paper.tex` (ICASSP template, 5 pages incl. references; `cd report && latexmk -pdf paper.tex`).
  `report/` has the style files; the full ICASSP author kit is local only (`docs/course/icassp2027_template/`, git-ignored).
  Figures show specimen cards only (F01-F05, B01-B03): `make_figures.ours()` asserts it.
- `run_glare_experiments.sh`: reruns all glare experiments after `prepare_data.py` (step `midv` = the external set).

## Conventions / gotchas
- Glare region for metrics = where the untreated image is > 40 grey levels brighter than the reference (`evaluate.glare_region`).
- Do-no-harm rule: a method may lower ssim_outside by at most 0.002 vs untreated (val/train).
- With n = 8-9 test images per condition, always use `compare_runs.py`, not differences of means.
  PaddleOCR churns a few words between near-identical images.
- SSIM rewards smoothing; it disagrees with OCR here. Report both.
- Current status (fixed-box metric): the effect depends on the engine.
  - DL fine-tuned clearly helps Tesseract on halo text (+0.22 on test).
  - Every method slightly hurts PaddleOCR in the halo, which reads the untreated halo at 0.87 already.
  - DL fine-tuned is far better on SSIM (see README "Findings").
  - MIDV-2020 (n = 100) replicates this: Tesseract halo +0.125 (DL ft), PaddleOCR halo −0.086; ft ≈ zero-shot there.
  - MIDV clips (multi-frame fusion): core 0.5 -> 0.99 (PaddleOCR), 0.2 -> 0.86 (Tesseract); best frame per field gets almost all of it.
  - Deblurring (Restormer ft) helps B_D (test n = 4: Tesseract +0.38) and B_G (n = 9: PaddleOCR +0.09, Tesseract +0.29); DL deglare after it adds
    +0.045 for Tesseract only. Order: no difference on B_G test. Wiener does not transfer (hurts sharp images and MIDV).
  - MIDV photos: Restormer alone is near neutral; Restormer -> DL gives Tesseract +0.086, PaddleOCR -0.059 (the DL step). Deblurring fused clips: no change.
- zsh does not word-split `$VAR` in commands: write arguments out (or use arrays) in one-off shell runs.
- Agreed plan rules (NEXT_STEPS "Decisions"; NEXT_STEPS is in priority order): no artificial/synthetic glare;
  the proposal's approach section is a guide, not a contract; new parameters are chosen on our train+val only;
  nothing is trained or tuned on MIDV; blur and glare share `split.csv` (test cards never used, in any condition).
