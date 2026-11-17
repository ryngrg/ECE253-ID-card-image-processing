# CLAUDE.md

ECE253 project (UCSD): improve ID-card photos for OCR. Two artifacts, one per teammate:
out-of-focus **blur** (teammate) and **glare** = specular highlights (this repo's glare code).
For each artifact: one classical method + one learning-based method (fine-tuned pretrained net),
compared with and without our modifications and against the untreated image. The plan is in `proposal.tex`;
current methods and results are in `README.md`; planned work is in `NEXT_STEPS.md`.

## Data (never delete anything in `data/raw/`)
- 141 PNGs of 22 cards (B01-B08, F01-F14). Name: `{card}_{B|S}_{G|D}_{num}.png`
  (B/S = blur/sharp, G/D = glare/diffuse).
- Reference for any glare image = the same card's `S_D_0`.
- Rectified, reference-aligned copies are in `data/rectified/*_rectified.png` (1012x638).
  `data/rectified/alignment_log.csv` holds the per-image status.
- `ground_truth/{card}.txt`: hand-typed card text (used for f1_gt / recall_gt).
- `split.csv`: card-level 14/4/4 train/val/test split. Never regenerate without `--resplit`.
  **Test is for final numbers only**: tune and select on train/val.
- `out/pairs.csv`: one row per glare image (S_G / B_G) with ref_path and split. Always filter `usable == True` (94 of 95).

## Environment
- venv: `card_scanning_env/bin/python` (Python 3.10, torch with MPS on an M2 Mac, no CUDA).
- Run every script from the repo root: `python src/<script>.py`. Scripts import each other from `src/`.
- `third_party/DocSHRNet` (code) and `third_party/weights/docshrnet.pth` are git-ignored; README has the download steps.
  Also git-ignored: `out/`, `checkpoints/`, `logs/`, `data/`, `results/ocr_cache/`.
- PaddleOCR (the engine we report) takes ~6.5 s/image on CPU, so a full evaluation of 94 images takes ~10 min.
  Results are cached in `results/ocr_cache/<engine>_<lang>/`, keyed by path + mtime: rewriting an output re-OCRs it.
  Run long jobs in the background.

## Scripts (`src/`)
- `rectify_ids.py`: card detection, warp to 1012x638, ECC homography alignment to the S_D reference.
- `prepare_data.py`: builds `out/pairs.csv` and `split.csv`, plus overlays in `out/overlays/`.
- `classical_glare.py`: two-zone classical method.
  - Clipped seeds, then a veil estimated by "statistics inpainting" (`veil_mode=stat`), white-level cap,
    background floor, and veil inversion on RGB.
  - Flat core fill (`core_fill=flat`; `telea` is the ablation).
  - `--tune` runs an SSIM grid with the do-no-harm rule.
  - Params: `classical_params.json` (final) and `classical_params_decay.json` (old distance-decay model, ablation).
  - Flags: `--no-veil`, `--no-inpaint`, `--core-fill`, `--debug`.
- `finetune_dl.py`: DocSHRNet backbone with a new detection head; output = m·removed + (1-m)·input.
  - Trains on train S_G; picks the best checkpoint on val S_G.
  - `--infer --checkpoint checkpoints/docshr_mt_best.pt` or `--infer --zero-shot`.
- `evaluate.py`: per-image metrics:
  - SSIM: whole image / glare region / outside the glare.
  - OCR: CER and word_f1 vs the reference's OCR, plus f1_gt / recall_gt vs the ground truth.
  - Usage: `--engine paddleocr --treated-dir out/X --name X`; `--combine` builds `results/all_methods.csv`.
  - Treated images must keep the rectified filenames.
- `compare_runs.py`: paired per-image differences with 95% bootstrap CIs (`results/compare_*.csv`).
- `analyze_ocr_loss.py`: share of reference words lost in the core / halo / outside zones.
- `ssim_diagnose.py`: why SSIM stays low outside the glare (fine detail, not misalignment).
- `run_glare_experiments.sh`: reruns all glare experiments after `prepare_data.py`.

## Conventions / gotchas
- Glare region for metrics = where the untreated image is > 40 grey levels brighter than the reference (`evaluate.glare_region`).
- Do-no-harm rule: a method may lower ssim_outside by at most 0.002 vs untreated (val/train).
- With n = 8-9 test images per condition, always use `compare_runs.py`, not differences of means.
  PaddleOCR churns a few words between near-identical images.
- SSIM rewards smoothing; it disagrees with OCR here. Report both.
- Current status: neither method significantly improves OCR on test. DL fine-tuned is far better on SSIM
  (see README "Findings").
