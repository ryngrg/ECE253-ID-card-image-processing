#!/usr/bin/env python3
"""
Paired comparison of two evaluate.py runs, image by image.

With only 8-9 images per split, a difference of 0.01 in the means is usually noise.  This
compares the SAME images under both runs, and reports for each metric the mean change, a
95% bootstrap confidence interval, and how many images got better / worse.  A change whose
interval includes 0 is not distinguishable from noise.

Usage
  python src/common/compare_runs.py --base untreated --new classical
  python src/common/compare_runs.py --base untreated --new classical --splits test
  python src/common/compare_runs.py --base untreated --new classical --worst 5
  python src/common/compare_runs.py --results-dir results/glare/box_paddle --base untreated --new dl_ft --splits test
      (fixed-box recognition per zone, from box_ocr.py: char_acc_* / exact_* columns)

Reads results/glare/whole_card/{name}_per_image.csv.  For cer, lower is better; the sign is flipped so that
a positive change always means "new is better".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["ssim", "ssim_glare", "ssim_outside", "cer", "word_f1", "f1_gt", "recall_gt"]
BOX_ZONES = ["core", "halo", "outside", "glare", "all", "digits_glare", "digits_all"]
METRICS += [f"{k}_{z}" for k in ("char_acc", "exact") for z in BOX_ZONES]  # box_ocr.py
LOWER_IS_BETTER = {"cer"}


def boot_ci(d: np.ndarray, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="run name, e.g. untreated")
    ap.add_argument("--new", required=True, help="run name, e.g. classical")
    ap.add_argument("--results-dir", default="results/glare/whole_card", type=Path)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--worst", type=int, default=0, help="also list the N worst images per metric")
    ap.add_argument("--metrics", default=None, help="comma-separated subset of metrics (default: all present)")
    a = ap.parse_args()
    metrics = a.metrics.split(",") if a.metrics else METRICS

    base = pd.read_csv(a.results_dir / f"{a.base}_per_image.csv")
    new = pd.read_csv(a.results_dir / f"{a.new}_per_image.csv")
    m = base.merge(new, on=["image", "card_id", "condition", "split"], suffixes=("_b", "_n"))
    m = m[m.split.isin(a.splits.split(","))]
    print(f"{len(m)} images in both runs.  Positive change = '{a.new}' is better than '{a.base}'.\n")

    rows = []
    for (split, cond), g in m.groupby(["split", "condition"]):
        for met in metrics:
            if f"{met}_b" not in g or g[f"{met}_b"].isna().all():
                continue
            d = (g[f"{met}_n"] - g[f"{met}_b"]).dropna().to_numpy()
            if met in LOWER_IS_BETTER:
                d = -d
            if len(d) < 2:
                continue
            lo, hi = boot_ci(d)
            rows.append(dict(split=split, condition=cond, metric=met, n=len(d), change=d.mean(),
                             ci_low=lo, ci_high=hi, better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()),
                             significant="yes" if lo > 0 or hi < 0 else "no"))
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    out = a.results_dir / f"compare_{a.base}_vs_{a.new}.csv"
    res.to_csv(out, index=False)

    if a.worst:
        for met in ["ssim_outside", "word_f1", "char_acc_glare"]:
            if f"{met}_b" not in m:
                continue
            d = (m[f"{met}_n"] - m[f"{met}_b"]) * (-1 if met in LOWER_IS_BETTER else 1)
            print(f"\nWorst {a.worst} images for {met} (new - base):")
            for i in d.nsmallest(a.worst).index:
                print(f"  {m.loc[i, 'image']:40s} {d[i]:+.3f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
