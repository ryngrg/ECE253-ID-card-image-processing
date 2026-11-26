#!/usr/bin/env bash
# Deblur alone and deblur + deglare chains (project root, venv active).  Needs the parameters / checkpoint of
# src/blur/README.md "Tune and train on our splits" and the glare outputs of run_glare_experiments.sh.
#   bash src/joint/run_chain_experiments.sh                         # everything (many hours: Restormer ~20 s/card on MPS)
#   STEPS="blur ours ours_eval" bash src/joint/run_chain_experiments.sh
# An output folder that exists is kept (deblur.py writes complete folders or nothing): delete it to redo a run.
#
# Run names:  A_B = first A, then B.   wiener / restormer = deblur only, classical / dl = deglare only (frozen glare methods)
#   wiener      fixed Wiener, params/blur/wiener_params.json          restormer      checkpoints/blur/restormer_ours.pt
#   wiener_ad   adaptive Wiener (the proposal's version)         restormer_pre  official pretrained weights
#   wiener_old  Wiener radius 3 / K 0.3 (before retuning)        dl             fine-tuned DocSHRNet
set -euo pipefail
PY=${PY:-python}
STEPS=${STEPS:-"blur blur_eval ours ours_eval midv midv_eval clips clips_eval"}
FT=${FT:-checkpoints/blur/restormer_ours.pt}
DLCK=checkpoints/glare/docshr_mt_best.pt
METRICS=char_acc_all,char_acc_core,char_acc_halo,char_acc_glare,char_acc_outside,char_acc_digits_all
has() { [[ " $STEPS " == *" $1 "* ]]; }

# deblur NAME IN OUT [extra deblur.py args]
deblur() {
  local m=$1 in=$2 out=$3; shift 3
  [ -d "$out" ] && { echo "[keep] $out"; return; }
  case $m in
    wiener)        $PY src/blur/deblur.py --method wiener --in-dir "$in" --out-dir "$out" "$@" ;;
    wiener_old)    $PY src/blur/deblur.py --method wiener --radius 3 --k 0.3 --in-dir "$in" --out-dir "$out" "$@" ;;
    wiener_ad)     $PY src/blur/deblur.py --method wiener_adaptive --in-dir "$in" --out-dir "$out" "$@" ;;
    restormer)     $PY src/blur/deblur.py --method restormer --weights "$FT" --in-dir "$in" --out-dir "$out" "$@" ;;
    restormer_pre) $PY src/blur/deblur.py --method restormer --in-dir "$in" --out-dir "$out" "$@" ;;
  esac | tail -1
}
# deglare NAME IN OUT <pairs args>
deglare() {
  local m=$1 in=$2 out=$3; shift 3
  [ -f "$out/.done" ] && { echo "[keep] $out"; return; }
  case $m in
    classical) $PY src/glare/classical_glare.py --params params/glare/classical_params.json --in-dir "$in" --out-dir "$out" "$@" ;;
    dl)        $PY src/glare/finetune_dl.py --infer --checkpoint $DLCK --in-dir "$in" --out-dir "$out" "$@" ;;
  esac | tail -1
  touch "$out/.done"
}
# box GROUP TAG PAIRS LAYOUT OUTROOT RUNS EXTRA SPLITS [flags]
box() {
  local group=$1 tag=$2 pairs=$3 layout=$4 root=$5 runs=$6 extra=$7 splits=$8; shift 8
  for eng in paddle tesseract; do
    $PY src/common/box_ocr.py --engine $eng --pairs "$pairs" --layout "$layout" --results-dir "results/$group" --tag "$tag" --out-root "$root" \
      --runs "$runs" --extra "$extra" --splits "$splits" "$@"
  done
}

# ---------------------------------------------------------------- blur-only images (B_D)
if has blur; then
  for m in wiener_old wiener wiener_ad restormer_pre restormer; do
    deblur $m data/rectified out/blur/$m --pairs out/pairs_blur.csv
  done
fi
if has blur_eval; then
  R=wiener_old,wiener,wiener_ad,restormer_pre,restormer
  box blur blur out/pairs_blur.csv results/layouts/box_layout.csv out/blur $R "" train,val,test
  $PY src/common/evaluate.py --no-ocr --pairs out/pairs_blur.csv --out-dir results/blur/ssim_blur --name untreated
  for m in ${R//,/ }; do
    $PY src/common/evaluate.py --no-ocr --pairs out/pairs_blur.csv --out-dir results/blur/ssim_blur --treated-dir out/blur/$m --name $m
  done
fi

# ---------------------------------------------------------------- our glare images (B_G; S_G = do-no-harm check)
if has ours; then
  O=out/chain; P="--pairs out/pairs.csv"
  for m in wiener wiener_ad restormer_pre restormer; do deblur $m data/rectified $O/$m $P; done
  # deblur -> deglare (cheap: the deblurred images exist)
  deglare classical $O/wiener        $O/wiener_classical
  deglare classical $O/wiener_ad     $O/wiener_ad_classical
  deglare dl        $O/restormer     $O/restormer_dl
  deglare classical $O/restormer     $O/restormer_classical
  # deglare -> deblur
  deblur wiener        out/classical   $O/classical_wiener  $P
  deblur wiener_ad     out/classical   $O/classical_wiener_ad $P
  deblur restormer     out/dl_ft       $O/dl_restormer      $P
fi
if has ours_eval; then
  R=wiener,wiener_ad,restormer_pre,restormer,wiener_classical,classical_wiener,restormer_dl,dl_restormer
  R=$R,wiener_ad_classical,classical_wiener_ad,restormer_classical
  box joint chain out/pairs.csv results/layouts/box_layout.csv out/chain $R \
    out/classical:classical,out/dl_ft:dl train,val,test
  $PY src/common/evaluate.py --no-ocr --out-dir results/joint/ssim_chain --name untreated
  for m in ${R//,/ }; do $PY src/common/evaluate.py --no-ocr --out-dir results/joint/ssim_chain --treated-dir out/chain/$m --name $m; done
  for x in classical:classical dl_ft:dl; do
    $PY src/common/evaluate.py --no-ocr --out-dir results/joint/ssim_chain --treated-dir out/${x%%:*} --name ${x##*:}
  done
fi

# ---------------------------------------------------------------- MIDV-2020 photos (external)
chain_midv() { # ROOT PAIRS SPLITS RECTIFIED
  local root=$1 pairs=$2 splits=$3 rect=$4
  local A="--pairs $pairs --splits $splits --conditions S_G"
  for m in wiener wiener_ad restormer; do deblur $m "$rect" $root/chain/$m --pairs "$pairs"; done
  deglare classical $root/chain/wiener    $root/chain/wiener_classical $A
  deglare classical $root/chain/wiener_ad $root/chain/wiener_ad_classical $A
  deglare dl        $root/chain/restormer $root/chain/restormer_dl     $A
  deblur wiener     $root/classical $root/chain/classical_wiener --pairs "$pairs"
  deblur wiener_ad  $root/classical $root/chain/classical_wiener_ad --pairs "$pairs"
  deblur restormer  $root/dl_ft     $root/chain/dl_restormer     --pairs "$pairs"
}
CH=wiener,wiener_ad,restormer,wiener_classical,classical_wiener,wiener_ad_classical,classical_wiener_ad,restormer_dl,dl_restormer
if has midv; then chain_midv out/midv out/pairs_midv.csv midv data/midv/rectified; fi
if has midv_eval; then
  box joint midv_chain out/pairs_midv.csv results/layouts/box_layout_midv.csv out/midv/chain $CH \
    out/midv/classical:classical,out/midv/dl_ft:dl midv --fold-accents --pad 2
fi

# ---------------------------------------------------------------- MIDV-2020 clips: deblurring of the fused images
# The single-image chains are not run here: the key frames passed the blur filter, and single-image deglare cannot
# restore their clipped text.  Only deblurring of the fused image is tested.
if has clips; then
  for m in wiener wiener_ad restormer; do
    deblur $m out/midv_clips/fused out/midv_clips/chain/fused_$m --pairs out/pairs_midv_clips.csv
  done
fi
if has clips_eval; then
  box joint midv_clips_chain out/pairs_midv_clips.csv results/layouts/box_layout_midv_clips.csv out/midv_clips/chain \
    fused_wiener,fused_wiener_ad,fused_restormer out/midv_clips/fused:fused keymid,keymax --fold-accents --pad 2
fi

# tables of every finished part -> results/{blur,joint}/*_{paired,pooled}.csv
$PY src/joint/summarise_chain.py
