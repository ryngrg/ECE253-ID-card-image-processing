#!/usr/bin/env bash
# Reproduce all glare experiments (run from the project root, with the venv active).
#   bash src/run_glare_experiments.sh            # everything
#   STEPS="eval compare" bash src/run_glare_experiments.sh
# Prerequisites: data/rectified/ (rectify_ids.py), out/pairs.csv (prepare_data.py),
#   third_party/DocSHRNet + third_party/weights/docshrnet.pth (see README).
set -euo pipefail
ENGINE=${ENGINE:-paddleocr}
STEPS=${STEPS:-"classical dl eval compare box tesseract midv clips"}
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
  $PY src/classical_glare.py --params classical_params.json --denoise guided --dn-h 8 --out-dir out/classical_dn
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

if has box; then
  # main OCR metric: fixed reference boxes, recognition only, scored per zone against the ground truth
  $PY src/box_ocr.py --engine paddle --runs classical,classical_veil,classical_inpaint,classical_telea,classical_decay,classical_dn,dl_zeroshot,dl_ft
  for m in classical classical_dn dl_zeroshot dl_ft reference ref_jitter; do
    $PY src/compare_runs.py --results-dir results/box_paddle --base untreated --new "$m" --splits test \
      --metrics char_acc_halo,char_acc_core,char_acc_outside,char_acc_glare,char_acc_digits_glare,exact_glare
  done
fi

if has tesseract; then
  # second OCR engine: whole card (--psm 11) and fixed line crops (--psm 7)
  $PY src/evaluate.py --engine tesseract --name untreated_tess
  for m in classical dl_zeroshot dl_ft; do
    $PY src/evaluate.py --engine tesseract --treated-dir "out/$m" --name "${m}_tess"
    $PY src/compare_runs.py --base untreated_tess --new "${m}_tess" --splits test --metrics cer,word_f1,f1_gt,recall_gt
  done
  $PY src/box_ocr.py --engine tesseract --runs classical,classical_dn,dl_zeroshot,dl_ft
  for m in classical classical_dn dl_zeroshot dl_ft reference ref_jitter; do
    $PY src/compare_runs.py --results-dir results/box_tesseract --base untreated --new "$m" --splits test \
      --metrics char_acc_halo,char_acc_core,char_acc_outside,char_acc_glare,char_acc_digits_glare,exact_glare
  done
fi

if has midv; then
  # external test set: MIDV-2020 "highlight present" photos (download steps in src/prepare_midv.py).
  # Nothing is tuned or trained on it: frozen classical_params.json and the frozen DL checkpoint.
  if [ -d data/external/midv2020/photo ]; then
    $PY src/prepare_midv.py
    MA="--pairs out/pairs_midv.csv --splits midv --conditions S_G"
    $PY src/classical_glare.py --params classical_params.json $MA --out-dir out/midv/classical --debug
    mkdir -p out/midv/dl_zeroshot out/midv/dl_ft
    $PY src/finetune_dl.py --infer --zero-shot $MA --out-dir out/midv/dl_zeroshot
    $PY src/finetune_dl.py --infer --checkpoint checkpoints/docshr_mt_best.pt $MA --out-dir out/midv/dl_ft
    for eng in paddle tesseract; do
      $PY src/box_ocr.py --engine $eng --pairs out/pairs_midv.csv --layout results/box_layout_midv.csv \
        --splits midv --tag midv --out-root out/midv --runs classical,dl_zeroshot,dl_ft --fold-accents --pad 2
      for m in classical dl_zeroshot dl_ft reference ref_jitter; do
        $PY src/compare_runs.py --results-dir results/box_${eng}_midv --base untreated --new "$m" --splits midv \
          --metrics char_acc_halo,char_acc_core,char_acc_outside,char_acc_glare,char_acc_all,char_acc_digits_glare
      done
    done
  else
    echo "[midv] data/external/midv2020/photo missing: see src/prepare_midv.py for the download"
  fi
fi

if has clips; then
  # multi-frame fusion on the MIDV-2020 highlight clips 90-99 (NEXT_STEPS step 5; clips.tar, see NEXT_STEPS).
  # Two key-frame rules = two "splits": keymid (middle frame) and keymax (most clipped core in the fields).
  if [ -d data/external/midv2020/clips/images ]; then
    $PY src/prepare_midv_clips.py --jobs 4
    $PY src/fuse_frames.py --jobs 3 --figure
    CA="--pairs out/pairs_midv_clips.csv --splits keymid,keymax --conditions S_G"
    $PY src/classical_glare.py --params classical_params.json $CA --out-dir out/midv_clips/classical
    $PY src/finetune_dl.py --infer --zero-shot $CA --out-dir out/midv_clips/dl_zeroshot
    $PY src/finetune_dl.py --infer --checkpoint checkpoints/docshr_mt_best.pt $CA --out-dir out/midv_clips/dl_ft
    for eng in paddle tesseract; do
      $PY src/box_ocr.py --engine $eng --pairs out/pairs_midv_clips.csv --layout results/box_layout_midv_clips.csv \
        --splits keymid,keymax --tag midv_clips --out-root out/midv_clips --fold-accents --pad 2 \
        --runs fused,median,best_frame,best_field,classical,dl_zeroshot,dl_ft
      for m in fused median best_frame best_field classical dl_zeroshot dl_ft reference ref_jitter; do
        $PY src/compare_runs.py --results-dir results/box_${eng}_midv_clips --base untreated --new "$m" \
          --splits keymid,keymax --metrics char_acc_core,char_acc_halo,char_acc_outside,char_acc_glare,char_acc_all
      done
      for b in best_frame best_field; do
        $PY src/compare_runs.py --results-dir results/box_${eng}_midv_clips --base "$b" --new fused \
          --splits keymid,keymax --metrics char_acc_core,char_acc_halo,char_acc_outside,char_acc_glare,char_acc_all
      done
    done
  else
    echo "[clips] data/external/midv2020/clips/images missing: see NEXT_STEPS step 5"
  fi
fi
