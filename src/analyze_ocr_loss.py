#!/usr/bin/env python3
"""
Where does the OCR loss on glare images come from?

Every word read on the glare-free reference is looked up in the OCR output of the glare image
(or of a processed image).  A reference word is "lost" if the other output has no matching word
left (multiset match, as in word F1).  Each word is placed in one zone by where its box sits:

  core     overlaps the clipped glare core: the information is destroyed
  halo     overlaps the rest of the glare region (brighter than the reference): washed out,
           potentially recoverable
  outside  elsewhere: OCR differences that have nothing to do with glare (noise floor)

Reading the result
  outside loss rate  = noise floor between two different photos; no glare method can fix it
  halo   loss rate well above outside  -> haze correction has something to gain
  core   loss rate high                -> mostly unrecoverable by classical inpainting
OCR results are cached (shared with evaluate.py), so only the first run is slow.
Run it on the untreated images, then with --treated-dir to see where a method gains or loses.

Usage
  python analyze_ocr_loss.py --engine paddleocr
  python analyze_ocr_loss.py --engine paddleocr --treated-dir out/classical --name classical
  python analyze_ocr_loss.py --engine paddleocr --conditions S_G,B_G --splits val,test

Word boxes: tesseract gives one box per word.  paddleocr / easyocr give one box per text line,
which is split among its words by character count, so zones are approximate.
"""
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classical_glare import DEFAULTS, detect_core
from evaluate import glare_region, make_ocr, normalize, ocr_file

OVERLAP = 0.10  # a word is in a zone if at least this fraction of its box overlaps it


def split_line_boxes(items):
    """Give each word its own box (items of one text line share the same box)."""
    groups = defaultdict(list)
    for t in items:
        groups[tuple(t[3])].append(t)
    words = []
    for box, g in groups.items():
        total = sum(len(t[2]) + 1 for t in g)
        x = box[0]
        for t in g:
            w = (box[2] - box[0]) * (len(t[2]) + 1) / total if len(g) > 1 else box[2] - box[0]
            words.append((t[2], (x, box[1], x + w, box[3])))
            x += w
    return words


def zone_of(box, core, glare):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    h, w = core.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return "outside"
    if core[y0:y1, x0:x1].mean() >= OVERLAP:
        return "core"
    if glare[y0:y1, x0:x1].mean() >= OVERLAP:
        return "halo"
    return "outside"


def report(res: pd.DataFrame, title: str):
    tab = res.groupby("zone").agg(ref_words=("found", "size"), lost=("found", lambda s: int((~s).sum())))
    tab["loss_rate"] = tab.lost / tab.ref_words
    tab["share_of_losses"] = tab.lost / tab.lost.sum()
    tab = tab.reindex([z for z in ("core", "halo", "outside") if z in tab.index])
    print(f"\n=== OCR word loss by zone: {title} ===")
    print(tab.round(3).to_string())
    n, lost = tab.ref_words.sum(), tab.lost.sum()
    if "outside" in tab.index and len(tab) > 1:
        r_out = tab.loc["outside", "loss_rate"]
        g = tab.drop(index="outside")
        excess = max(0.0, g.lost.sum() - (g.ref_words * r_out).sum())
        print(f"overall loss rate {lost / n:.3f}.  If the glare zones were read as well as the rest of the "
              f"card ({r_out:.3f}) it would be {(lost - excess) / n:.3f}: glare accounts for ~{excess:.0f} of "
              f"{lost} lost words ({excess / lost:.0%}); this is the most any glare method can win back.")
    else:
        print(f"overall loss rate {lost / n:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--treated-dir", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--engine", default="tesseract", choices=["tesseract", "paddleocr", "easyocr"])
    ap.add_argument("--lang", default=None)
    ap.add_argument("--conditions", default="S_G")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--no-cache", action="store_true", help="do not reuse stored OCR results")
    a = ap.parse_args()
    cache_dir = None if a.no_cache else a.out_dir / "ocr_cache" / f"{a.engine}_{a.lang or 'default'}"

    name = a.name or (a.treated_dir.name if a.treated_dir else "untreated")
    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs.usable & pairs.condition.isin(a.conditions.split(","))
                  & pairs.split.isin(a.splits.split(","))].reset_index(drop=True)
    ocr = make_ocr(a.engine, a.lang)
    print(f"'{name}': {len(pairs)} images ({a.conditions}; {a.splits})")

    ref_cache, rows = {}, []
    t0 = time.time()
    for k, r in enumerate(pairs.itertuples(), 1):
        ref = cv2.imread(r.ref_path)
        untreated = cv2.imread(r.path)
        if r.ref_path not in ref_cache:
            ref_cache[r.ref_path] = split_line_boxes(ocr_file(ocr, r.ref_path, cache_dir))
        ref_words = ref_cache[r.ref_path]

        img_path = a.treated_dir / Path(r.path).name if a.treated_dir else Path(r.path)
        img = cv2.imread(str(img_path))
        if img is None or img.shape != ref.shape:
            print(f"[warn] skipping {Path(r.path).name}")
            continue
        hyp = Counter(normalize(t[2] for t in ocr_file(ocr, img_path, cache_dir)))

        core = detect_core(untreated, DEFAULTS)
        glare = glare_region(cv2.cvtColor(untreated, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        for text, box in ref_words:
            for w in normalize([text]):
                found = hyp[w] > 0
                if found:
                    hyp[w] -= 1
                rows.append(dict(image=Path(r.path).name, card_id=r.card_id, condition=r.condition, word=w,
                                 zone=zone_of(box, core, glare), found=found))
        el = time.time() - t0
        print(f"  {k}/{len(pairs)}  {el:.0f}s elapsed, ~{el / k * (len(pairs) - k):.0f}s left", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(a.out_dir / f"ocr_loss_{name}.csv", index=False)

    conds = sorted(res.condition.unique())
    if len(conds) > 1:
        for c in conds:
            report(res[res.condition == c], f"{name} ({a.engine}), {c} only")
    report(res, f"{name} ({a.engine}), {' + '.join(conds)}")
    digits = res[res.word.str.contains(r"\d")]
    if len(digits):
        report(digits, f"{name}: only words containing digits (dates, ID numbers)")
    lost_core = res[(res.zone == "core") & ~res.found].word.tolist()
    if lost_core:
        print(f"\nwords lost in the core (first 25): {lost_core[:25]}")
    print(f"Saved {a.out_dir}/ocr_loss_{name}.csv")


if __name__ == "__main__":
    main()
