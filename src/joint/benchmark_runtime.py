#!/usr/bin/env python3
"""
Runtime per image of each glare and deblurring method, measured the same way for all of them.

What is timed: only the processing call of each method on an image already in memory, i.e. exactly
what the batch scripts call per image, without disk I/O:
  classical      classical_glare.remove_glare(img, params/glare/classical_params.json)        CPU (OpenCV / NumPy)
  dl_zeroshot    finetune_dl.predict(...) with the pretrained DocSHRNet, no gate  MPS and CPU
  dl_ft          finetune_dl.predict(...) with checkpoints/glare/docshr_mt_best.pt       MPS and CPU
  wiener / wiener_adaptive  deblur_blocks.DeblurBlock with the params/blur/*.json settings     CPU
                 (the adaptive one includes its 9 Tesseract calls)
  restormer      DeblurBlock with checkpoints/blur/restormer_ours.pt (pretrained weights if missing), tiled  MPS and CPU
  predict() includes the host -> device copy, padding to a multiple of 16, the forward pass and the copy
  back (its .cpu() call synchronises MPS, so the time is the real wall time).
Protocol: `--warmup` images first (not scored: the first MPS calls compile kernels), then every image
`--repeats` times (`--cpu-dl-repeats` for the DL methods on CPU, ~10 s per call); the per-image time is the median over repeats, and the summary gives the median /
mean / p90 over images.  The classical method returns early on images with no clipped seed, so its
median is reported separately for images with and without seeds.

Usage
  python src/joint/benchmark_runtime.py                                   # our test S_G + B_G (17 images, 1012x638)
  python src/joint/benchmark_runtime.py --pairs out/pairs_midv.csv --splits midv --conditions S_G \\
      --max-images 20 --name midv                                    # MIDV (cards 1012 px, passports 1478 px)
  python src/joint/benchmark_runtime.py --methods wiener,wiener_adaptive,restormer --name deblur_test   # deblurring
Writes results/runtime/runtime_{name}.csv (summary) and results/runtime/runtime_{name}_per_image.csv.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from glare.classical_glare import DEFAULTS, detect_core, remove_glare
from glare import finetune_dl as fdl


def time_call(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def dl_model(kind: str, checkpoint: str):
    """Build the DL model exactly as finetune_dl.py --infer does."""
    a = argparse.Namespace(model="docshrnet", model_repo=None, model_kwargs=None, pretrained=None,
                           no_gate=False)
    if kind == "dl_zeroshot":
        a.no_gate = True
        return fdl.build_model(a)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    a.no_gate = saved.get("no_gate", False)
    a.model = saved.get("model", "docshrnet")
    model = fdl.build_model(a)
    model.load_state_dict(ck["state_dict"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--conditions", default="S_G,B_G")
    ap.add_argument("--max-images", type=int, default=0, help="evenly spaced subset (0 = all)")
    ap.add_argument("--params", default="params/glare/classical_params.json")
    ap.add_argument("--checkpoint", default="checkpoints/glare/docshr_mt_best.pt")
    ap.add_argument("--devices", default="mps,cpu", help="devices for the DL methods (unavailable ones are skipped)")
    ap.add_argument("--methods", default="classical,dl_zeroshot,dl_ft")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu-dl-repeats", type=int, default=1,
                    help="repeats for the DL methods on CPU (~10 s per call; single calls vary < 1%%)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--name", default="test")
    ap.add_argument("--results-dir", default="results/runtime", type=Path)
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs.usable & pairs.split.isin(a.splits.split(",")) &
                  pairs.condition.isin(a.conditions.split(","))].reset_index(drop=True)
    if a.max_images and len(pairs) > a.max_images:
        pairs = pairs.iloc[np.linspace(0, len(pairs) - 1, a.max_images).round().astype(int)].reset_index(drop=True)
    imgs = [cv2.imread(p) for p in pairs.path]
    names = [Path(p).name for p in pairs.path]
    p_cls = {**DEFAULTS, **json.loads(Path(a.params).read_text())}
    seeds = [bool(detect_core(im, p_cls).any()) for im in imgs]
    print(f"{len(imgs)} images ({a.splits} {a.conditions}), sizes "
          f"{sorted({f'{im.shape[1]}x{im.shape[0]}' for im in imgs})}, {sum(seeds)} with clipped seeds")
    print(f"{platform.platform()}, {platform.processor()}, torch {torch.__version__}, "
          f"{torch.get_num_threads()} CPU threads\n", flush=True)

    jobs = []  # (method, device, fn(img) -> None)
    methods = a.methods.split(",")
    if "classical" in methods:
        jobs.append(("classical", "cpu", lambda im: remove_glare(im, p_cls)))
    for kind in ("dl_zeroshot", "dl_ft"):
        if kind not in methods:
            continue
        for dev in a.devices.split(","):
            if dev == "mps" and not torch.backends.mps.is_available():
                print("[skip] MPS not available")
                continue
            model = dl_model(kind, a.checkpoint).to(dev).eval()
            d = torch.device(dev)
            jobs.append((kind, dev, lambda im, m=model, d=d:
                         fdl.predict(m, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), d, 16)))

    from blur.deblur_blocks import DeblurBlock
    if "wiener" in methods:
        c = json.loads(Path("params/blur/wiener_params.json").read_text())
        jobs.append(("wiener", "cpu", DeblurBlock("wiener", radius=c["radius"], k=c["k"], color_order="bgr")))
    if "wiener_adaptive" in methods:
        f = Path("params/blur/wiener_adaptive_params.json")
        c = {k: v for k, v in json.loads(f.read_text()).items() if k != "description"} if f.exists() else {}
        jobs.append(("wiener_adaptive", "cpu", DeblurBlock("wiener_adaptive", adaptive=c, color_order="bgr")))
    if "restormer" in methods:
        w = Path("checkpoints/blur/restormer_ours.pt")
        for dev in a.devices.split(","):
            jobs.append(("restormer", dev, DeblurBlock("restormer", weights=w if w.exists() else None, device=dev,
                                                       color_order="bgr")))

    rows = []
    for method, dev, fn in jobs:
        reps = a.cpu_dl_repeats if (dev == "cpu" and method in ("dl_zeroshot", "dl_ft", "restormer")) else a.repeats
        for im in imgs[:a.warmup]:
            fn(im)
        t_all = time.perf_counter()
        for name, im, sd in zip(names, imgs, seeds):
            rows.append(dict(method=method, device=dev, image=name, width=im.shape[1], height=im.shape[0],
                             has_seeds=sd, seconds=time_call(lambda: fn(im), reps)))
        r = pd.DataFrame([x for x in rows if x["method"] == method and x["device"] == dev])
        print(f"{method:12s} {dev:4s} median {r.seconds.median():.3f} s/image  "
              f"(p90 {r.seconds.quantile(0.9):.3f}; {time.perf_counter() - t_all:.0f}s total)", flush=True)
        if dev == "mps":
            torch.mps.empty_cache()

    per = pd.DataFrame(rows)
    summ = []
    for (m, d), g in per.groupby(["method", "device"], sort=False):
        row = dict(method=m, device=d, n=len(g), median_s=g.seconds.median(), mean_s=g.seconds.mean(),
                   p90_s=g.seconds.quantile(0.9), max_s=g.seconds.max())
        if m == "classical":
            row["median_s_with_seeds"] = g[g.has_seeds].seconds.median()
            row["median_s_no_seeds"] = g[~g.has_seeds].seconds.median()
        summ.append(row)
    summ = pd.DataFrame(summ)
    a.results_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(a.results_dir / f"runtime_{a.name}_per_image.csv", index=False)
    summ.to_csv(a.results_dir / f"runtime_{a.name}.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False))
    print(f"\nSaved {a.results_dir}/runtime_{a.name}.csv")


if __name__ == "__main__":
    main()
