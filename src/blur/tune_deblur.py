#!/usr/bin/env python3
"""
Choose the deblurring parameters on our train / val B_D images (out/pairs_blur.csv), never on test.

Port of the blur repo's selection (scripts/run_real_mixed.py + scripts/real_mixed_metrics.py and
wiener/weiner_training.py; Mlazazarus/ECE253 @ 2893d66) with this repo's metric: fixed-box character accuracy
(box_ocr.py; pooled over all labelled boxes, mean over the engines) and whole-image SSIM vs S_D.

  wiener            radius x K grid of the fixed filter (train + val: nothing is learned, so the pool is larger)
  wiener_adaptive   noise_scale / sharpness / ringing weights of the adaptive filter (train + val)
  restormer         pretrained weights and every step_*.pt of --ckpt-dir (val only: it was trained on train)

Rule (fixed in advance, same as the blur repo): among the candidates that improve BOTH character accuracy and
SSIM over the untreated images, the highest accuracy (tie: SSIM).  If there is none, the smallest sum of
the accuracy and SSIM ranks; the output then says that no candidate improves both.
Radii are in rectified pixels.  Readings are cached in results/ocr_cache/tune_{engine}/ by pixel content, so
an interrupted run resumes.

Writes results/blur/tune_deblur/{method}.csv (every candidate) and {method}_selection.json.  --write also
updates params/blur/wiener_params.json / params/blur/wiener_adaptive_params.json, or copies the chosen checkpoint to
checkpoints/blur/restormer_ours.pt (unless the pretrained weights win).

Usage
  python src/blur/tune_deblur.py wiener --write
  python src/blur/tune_deblur.py wiener_adaptive --write
  python src/blur/tune_deblur.py restormer --ckpt-dir checkpoints/blur/restormer_ours --write
  (add  --engines tesseract --limit 2  for a quick check; --limit runs never write parameters)
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common import privacy
from common.box_ocr import crop, make_recogniser, norm
from blur.deblur_blocks import DeblurBlock, adaptive_wiener, digest
from blur.deblur_blocks.wiener import restore
from common.evaluate import edit_distance
from blur.finetune_deblur import load_pairs

RADII = [1, 2, 3, 4, 5, 6, 8, 10]
KS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
NOISE_SCALES = [1, 10, 100, 1000]
SHARPNESS = [0, 0.25, 1]
RINGING = [0.25, 1, 4]
ROOT = Path(__file__).resolve().parents[2]


class Scorer:
    """Per-image character edits (per engine) and SSIM of a uint8 RGB image."""

    def __init__(self, engines, cache_root):
        self.engines, self.rec = engines, {e: make_recogniser(e) for e in engines}
        self.cache = {e: cache_root / f"tune_{e}" for e in engines}
        for d in self.cache.values():
            d.mkdir(parents=True, exist_ok=True)

    def read(self, engine, bgr, boxes):
        coords = boxes[["x0", "y0", "x1", "y1"]].to_numpy()
        key = hashlib.sha1(bgr.tobytes() + coords.round(1).tobytes()).hexdigest()
        f = self.cache[engine] / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        texts = self.rec[engine]([crop(bgr, b) for b in coords])
        f.write_text(json.dumps(texts))
        return texts

    def __call__(self, sample, rgb):
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        out = dict(image=sample["name"], tlen=int(sample["boxes"].target.str.len().sum()),
                   ssim=float(structural_similarity(sample["ref"], cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), data_range=255)))
        for e in self.engines:
            texts = self.read(e, bgr, sample["boxes"])
            out[f"ed_{e}"] = int(sum(edit_distance(t, norm(h)) for t, h in zip(sample["boxes"].target, texts)))
        return out


def load_samples(a, splits):
    layout = privacy.read_csv(a.layout, keep_default_na=False)  # targets of the real cards: private/ copy
    layout = layout[layout.labelled.astype(str) == "True"]
    samples = []
    for split in splits:
        for r in load_pairs(a.pairs, split).itertuples():
            b = layout[layout.card_id == r.card_id]
            if b.empty:
                print(f"[warn] {r.path}: no labelled boxes, skipped")
                continue
            samples.append(dict(name=Path(r.path).name, split=split, boxes=b,
                                input=cv2.imread(r.path)[..., ::-1].copy(), ref=cv2.imread(r.ref_path, cv2.IMREAD_GRAYSCALE)))
    return samples[:a.limit] if a.limit else samples


def summarise(rows, engines):
    d = pd.DataFrame(rows)
    out = dict(n=len(d), ssim=d.ssim.mean(), **{f"char_acc_{e}": 1 - d[f"ed_{e}"].sum() / d.tlen.sum() for e in engines})
    out["char_acc"] = float(np.mean([out[f"char_acc_{e}"] for e in engines]))
    return out


def select(table):
    """table: one row per candidate, first row = untreated.  Returns (index, improves_both)."""
    base, cand = table.iloc[0], table.iloc[1:]
    joint = cand[(cand.char_acc > base.char_acc) & (cand.ssim > base.ssim)]
    if len(joint):
        return joint.sort_values(["char_acc", "ssim"], ascending=False, kind="stable").index[0], True
    rank = cand.char_acc.rank(ascending=False, method="min") + cand.ssim.rank(ascending=False, method="min")
    return cand.assign(rank_sum=rank).sort_values(["rank_sum", "char_acc"], ascending=[True, False], kind="stable").index[0], False


def run_candidates(scorer, samples, cands, jobs):
    """cands: list of (settings dict, fn(sample) -> uint8 RGB)."""
    table, t0 = [], time.time()
    # Tesseract runs in subprocesses, so threads help; the Paddle recogniser is not thread-safe.
    workers = jobs if scorer.engines == ["tesseract"] else 1
    with ThreadPoolExecutor(workers) as pool:
        for i, (settings, fn) in enumerate(cands):
            rows = list(pool.map(lambda s: scorer(s, fn(s)), samples))
            table.append({**settings, **summarise(rows, scorer.engines)})
            print(f"[{i + 1}/{len(cands)}] {settings}  char_acc {table[-1]['char_acc']:.4f}  ssim {table[-1]['ssim']:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(table)


def to_float(sample):
    return sample["input"].astype(np.float32) / 255


def tune_wiener(a, scorer, samples):
    cands = [(dict(candidate="untreated", radius=0, k=0.0), lambda s: s["input"])]
    for r in RADII:
        for k in KS:
            cands.append((dict(candidate=f"r{r}_k{k:g}", radius=r, k=k),
                          lambda s, r=r, k=k: adaptive_wiener.to_uint8(restore(to_float(s), r, k))))
    return run_candidates(scorer, samples, cands, a.jobs)


def tune_adaptive(a, scorer, samples):
    """Every (noise scale, radius) image is made and scored once; the policies only choose among them."""
    max_radius = adaptive_wiener.DEFAULTS["max_radius"]
    cache = {}
    for i, s in enumerate(samples):
        for scale in NOISE_SCALES:
            cache[i, scale] = [(f, scorer(s, img)) for img, f in adaptive_wiener.candidates(to_float(s), scale, max_radius)]
        print(f"prepared {i + 1}/{len(samples)} {s['name']}", flush=True)
    table = [dict(candidate="untreated", **summarise([scorer(s, s["input"]) for s in samples], scorer.engines))]
    for scale in NOISE_SCALES:
        for sharp in SHARPNESS:
            for ring in RINGING:
                p = dict(max_radius=max_radius, noise_scale=scale, sharpness_weight=sharp, ringing_weight=ring)
                picks = [adaptive_wiener.choose([f for f, _ in cache[i, scale]], p) for i in range(len(samples))]
                rows = [cache[i, scale][j][1] for i, j in enumerate(picks)]
                table.append(dict(candidate=f"n{scale}_s{sharp:g}_r{ring:g}", **p, mean_radius=float(np.mean(picks)),
                                  **summarise(rows, scorer.engines)))
    return pd.DataFrame(table)


def tune_restormer(a, scorer, samples):
    from blur.deblur_blocks.restormer import DEFAULT_WEIGHTS
    paths = [DEFAULT_WEIGHTS, *sorted(a.ckpt_dir.glob("step_*.pt"))]
    if len(paths) == 1:
        raise SystemExit(f"No step_*.pt in {a.ckpt_dir}: run finetune_deblur.py first.")
    tables = [run_candidates(scorer, samples, [(dict(candidate="untreated", weights=""), lambda s: s["input"])], 1)]
    for p in paths:
        block = DeblurBlock("restormer", weights=p, device=a.device)
        check_training_cards(block, p)
        name = "pretrained" if p == DEFAULT_WEIGHTS else p.stem
        tables.append(run_candidates(scorer, samples, [(dict(candidate=name, weights=str(p)),
                                                        lambda s, b=block: b(s["input"]))], 1))
    return pd.concat(tables, ignore_index=True)


def check_training_cards(block, path):
    """Refuse checkpoints that were not trained on exactly our train cards."""
    meta = block.apply.provenance["fine_tuning"]
    if meta is None:
        return
    split = pd.read_csv("split.csv").set_index("card_id").split
    cards = meta.get("train_cards")
    if cards is None or any(split.get(c) != "train" for c in cards):
        raise SystemExit(f"{path}: not trained with our split (src/blur/finetune_deblur.py); it may have seen val / test cards.")


def write_params(a, method, best):
    def dump(name, obj):
        (ROOT / name).write_text(json.dumps(obj, indent=2) + "\n")
        print(f"updated {name}")

    note = f"Chosen by src/blur/tune_deblur.py {method} on our train + val B_D images. Radius is in rectified pixels."
    if method == "wiener":
        dump("params/blur/wiener_params.json", dict(radius=int(best.radius), k=float(best.k), description=note))
    elif method == "wiener_adaptive":
        dump("params/blur/wiener_adaptive_params.json",
             {k: int(best[k]) if k == "max_radius" else float(best[k]) for k in adaptive_wiener.DEFAULTS} | {"description": note})
    elif best.candidate == "pretrained":
        print("the pretrained weights win: nothing to copy")
    else:
        dst = ROOT / "checkpoints/blur/restormer_ours.pt"
        shutil.copyfile(best.weights, dst)
        print(f"copied {best.weights} -> {dst.relative_to(ROOT)}  sha256 {digest(dst)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["wiener", "wiener_adaptive", "restormer"])
    ap.add_argument("--pairs", default="out/pairs_blur.csv", type=Path)
    ap.add_argument("--layout", default="results/layouts/box_layout.csv", type=Path)
    ap.add_argument("--engines", default="paddle,tesseract")
    ap.add_argument("--ckpt-dir", default="checkpoints/blur/restormer_ours", type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--jobs", type=int, default=4, help="threads (used only with --engines tesseract)")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--write", action="store_true", help="update the parameter file / copy the checkpoint")
    ap.add_argument("--results-dir", default="results/blur", type=Path)
    ap.add_argument("--cache-dir", default="results/ocr_cache", type=Path)
    a = ap.parse_args()
    if a.limit and a.write:
        ap.error("--limit is a smoke test: it cannot --write parameters.")

    samples = load_samples(a, ["val"] if a.method == "restormer" else ["train", "val"])
    scorer = Scorer(a.engines.split(","), a.cache_dir)
    print(f"{len(samples)} images, {sum(len(s['boxes']) for s in samples)} boxes, engines {a.engines}", flush=True)
    table = {"wiener": tune_wiener, "wiener_adaptive": tune_adaptive, "restormer": tune_restormer}[a.method](a, scorer, samples)
    i, joint = select(table)
    best, base = table.loc[i], table.iloc[0]

    out = a.results_dir / "tune_deblur"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.method}_limit{a.limit}" if a.limit else a.method
    table.to_csv(out / f"{tag}.csv", index=False)
    selection = dict(method=a.method, improves_both=joint, engines=a.engines, images=[s["name"] for s in samples],
                     split_sha256=digest("split.csv"), selected=json.loads(best.to_json()),
                     untreated=json.loads(base.to_json()))
    (out / f"{tag}_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"\nselected {best.candidate}: char_acc {base.char_acc:.4f} -> {best.char_acc:.4f}, "
          f"ssim {base.ssim:.4f} -> {best.ssim:.4f}"
          + ("" if joint else "\n[warn] no candidate improves both char_acc and SSIM over the untreated images"))
    if a.write:
        write_params(a, a.method, best)


if __name__ == "__main__":
    main()
