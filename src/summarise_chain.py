#!/usr/bin/env python3
"""
Tables for the deblur-only and deblur + deglare experiments (run_chain_experiments.sh).

For each finished part (results/box_{engine}_{blur,chain,midv_chain,midv_clips_chain}/) writes to results/chain/:
  {part}_pooled.csv   char accuracy pooled over the boxes of each zone (box_ocr summary.csv), per split / condition
  {part}_paired.csv   paired per-image differences with 95% bootstrap CIs (as compare_runs.py), incl. SSIM if it
                      was computed (results/ssim_{blur,chain}/): every run vs untreated, and for a chain A_B also
                      vs A alone, vs B alone, and vs the reverse order B_A
Several splits are also pooled into one group "train+val" (consistency check; the DL models saw the train cards).

Usage:  python src/summarise_chain.py [--results-dir results]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_runs import boot_ci

PARTS = {"blur": "ssim_blur", "chain": "ssim_chain", "midv_chain": None, "midv_clips_chain": None}
SINGLE = ["wiener_old", "wiener_ad", "wiener", "restormer_pre", "restormer", "classical", "dlzs", "dl", "fused"]
METRICS = [f"char_acc_{z}" for z in ("all", "core", "halo", "glare", "outside", "digits_all")] + ["ssim", "ssim_glare", "ssim_outside"]
KEYS = ["image", "card_id", "condition", "split"]


def parse_chain(name):
    """'restormer_pre_dlzs' -> ('restormer_pre', 'dlzs'); None for a single method."""
    if name in SINGLE:
        return None
    for a in SINGLE:
        if name.startswith(a + "_") and name[len(a) + 1:] in SINGLE:
            return a, name[len(a) + 1:]
    return None


def bases(name, runs):
    out = ["untreated"]
    c = parse_chain(name)
    if c:
        out += [b for b in (*c, f"{c[1]}_{c[0]}") if b in runs]
    return out


def paired(tables, engine):
    rows = []
    for new, n in tables.items():
        if new in ("untreated", "ref_jitter"):
            continue
        for base in bases(new, tables):
            m = tables[base].merge(n, on=KEYS, suffixes=("_b", "_n"))
            groups = [((s, c), g) for (s, c), g in m.groupby(["split", "condition"])]
            groups += [(("train+val", c), g) for c, g in m[m.split.isin(["train", "val"])].groupby("condition")]
            for (split, cond), g in groups:
                for met in METRICS:
                    if f"{met}_b" not in g:
                        continue
                    d = (g[f"{met}_n"] - g[f"{met}_b"]).dropna().to_numpy()
                    if len(d) < 2:
                        continue
                    lo, hi = boot_ci(d)
                    rows.append(dict(engine=engine, split=split, condition=cond, new=new, base=base, metric=met, n=len(d),
                                     mean_base=g[f"{met}_b"].mean(), mean_new=g[f"{met}_n"].mean(), change=d.mean(),
                                     ci_low=lo, ci_high=hi, better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()),
                                     significant="yes" if lo > 0 or hi < 0 else "no"))
    return rows


def load_runs(folder):
    return {f.name[:-len("_per_image.csv")]: pd.read_csv(f) for f in sorted(folder.glob("*_per_image.csv"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", type=Path)
    a = ap.parse_args()
    out = a.results_dir / "chain"
    out.mkdir(parents=True, exist_ok=True)
    for part, ssim_dir in PARTS.items():
        rows, pooled = [], []
        for engine in ("paddle", "tesseract"):
            d = a.results_dir / f"box_{engine}_{part}"
            if not (d / "summary.csv").exists():
                continue
            pooled.append(pd.read_csv(d / "summary.csv"))
            rows += paired(load_runs(d), engine)
        if ssim_dir and (a.results_dir / ssim_dir).is_dir():
            rows += paired(load_runs(a.results_dir / ssim_dir), "ssim")
        if not rows:
            print(f"[skip] {part}: no results")
            continue
        pd.DataFrame(rows).to_csv(out / f"{part}_paired.csv", index=False)
        if pooled:
            pd.concat(pooled).to_csv(out / f"{part}_pooled.csv", index=False)
        print(f"{part}: {len(rows)} paired comparisons -> {out}/{part}_paired.csv")


if __name__ == "__main__":
    main()
