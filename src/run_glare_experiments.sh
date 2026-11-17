#!/usr/bin/env bash
# Reproduce all glare experiments (run from the project root, with the venv active).
#   bash src/run_glare_experiments.sh            # everything
#   STEPS="eval compare" bash src/run_glare_experiments.sh
# Prerequisites: data/rectified/ (rectify_ids.py), out/pairs.csv (prepare_data.py),
#   third_party/DocSHRNet + third_party/weights/docshrnet.pth (see README).
set -euo pipefail
ENGINE=${ENGINE:-paddleocr}
STEPS=${STEPS:-"classical dl eval compare"}
PY=${PY:-python}

has() { [[ " $STEPS " == *" $1 "* ]]; }

if has classical; then
  # classical_params.json: final parameters (chosen by OCR on train+val, see README);
  # classical_params_decay.json: the earlier distance-decay model with its earlier tuned values.
  # (python src/classical_glare.py --tune ... re-runs the SSIM grid search if wanted)
  $PY src/classical_glare.py --params classical_params.json --out-dir out/classical --debug
  $PY src/classical_glare.py --params classical_params.json --no-veil   --out-dir out/classical_inpaint
  $PY src/classical_glare.py --params classical_params.json --no-inpaint --out-dir out/classical_veil
  $PY src/classical_glare.py --params classical_params.json --core-fill telea --out-dir out/classical_telea
  $PY src/classical_glare.py --params classical_params_decay.json --out-dir out/classical_decay
fi

if has dl; then
  [ -f checkpoints/docshr_mt_best.pt ] || $PY src/finetune_dl.py --name docshr_mt --iters 1500 --val-every 150
  $PY src/finetune_dl.py --infer --zero-shot --out-dir out/dl_zeroshot
  $PY src/finetune_dl.py --infer --checkpoint checkpoints/docshr_mt_best.pt --out-dir out/dl_ft
fi

if has eval; then
  $PY src/evaluate.py --engine "$ENGINE"
  for m in classical classical_inpaint classical_veil classical_telea classical_decay dl_zeroshot dl_ft; do
    [ -d "out/$m" ] && $PY src/evaluate.py --engine "$ENGINE" --treated-dir "out/$m" --name "$m"
  done
  $PY src/evaluate.py --combine
fi

if has compare; then
  for m in classical classical_decay dl_zeroshot dl_ft; do
    [ -f "results/${m}_per_image.csv" ] && $PY src/compare_runs.py --base untreated --new "$m" --splits test
  done
  $PY src/compare_runs.py --base classical --new dl_ft --splits test
  $PY src/analyze_ocr_loss.py --engine "$ENGINE"
  for m in classical dl_ft; do
    $PY src/analyze_ocr_loss.py --engine "$ENGINE" --treated-dir "out/$m" --name "$m"
  done
fi
