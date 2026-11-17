#!/usr/bin/env python3
"""
Why is SSIM low even away from the glare?  Measures how much of the gap each
suspect explains, on the untreated glare images, using only pixels far from the glare.

Variants of SSIM(reference, image), all computed outside the glare region:
  raw         as evaluate.py computes it
  photo       after fitting a global gain/offset of the image to the reference
              (exposure / white-balance differences between the two shots)
  shift       after the best integer shift in +-max_shift px (residual misalignment)
              on top of photo; prints the best (dx, dy)
  blur        both images lightly blurred first (fine-detail differences: noise,
              sharpness, sub-pixel misalignment)
  all         blur + photo + shift

How to read it
  shift >> raw, best (dx,dy) not ~(0,0)      -> alignment is the problem
  photo >> raw                               -> exposure/colour differences
  blur  >> raw                               -> fine detail (noise, sharpness, subpixel);
                                                NB blurring also forgives small misalignment,
                                                so judge it together with the shift column
  none of them helps                         -> real appearance differences
                                                (holograms, lighting, glossy surface)

Usage
  python ssim_diagnose.py                       # 20 random S_G images
  python ssim_diagnose.py --n 0 --condition B_G # all B_G images
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from evaluate import glare_region

CROP = 6  # px ignored at the border so shifted images are comparable


def ssim_outside(ref, img, outside):
    _, smap = structural_similarity(ref, img, data_range=255, full=True)
    return float(smap[CROP:-CROP, CROP:-CROP][outside[CROP:-CROP, CROP:-CROP]].mean())


def photometric_match(ref, img, outside):
    x = img[outside][::7].astype(np.float32)
    y = ref[outside][::7].astype(np.float32)
    a, b = np.polyfit(x, y, 1)
    return np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)


def shift_img(img, dx, dy):
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def best_shift(ref, img, outside, max_shift):
    best = (-1.0, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            s = ssim_outside(ref, shift_img(img, dx, dy), outside)
            if s > best[0]:
                best = (s, dx, dy)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--condition", default="S_G")
    ap.add_argument("--n", type=int, default=20, help="random subset size (0 = all)")
    ap.add_argument("--max-shift", type=int, default=3)
    ap.add_argument("--blur", type=float, default=1.5, help="sigma for the blur variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ssim_diagnose.csv", type=Path)
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs.usable & (pairs.condition == a.condition)].reset_index(drop=True)
    if a.n and a.n < len(pairs):
        pairs = pairs.sample(a.n, random_state=a.seed).reset_index(drop=True)
    print(f"{len(pairs)} {a.condition} images")

    rows = []
    for k, r in enumerate(pairs.itertuples(), 1):
        ref = cv2.cvtColor(cv2.imread(r.ref_path), cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2GRAY)
        if ref.shape != img.shape:
            continue
        glare = glare_region(img, ref)
        outside = ~cv2.dilate(glare.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        if outside.sum() < 1000:
            continue

        raw = ssim_outside(ref, img, outside)
        photo_img = photometric_match(ref, img, outside)
        photo = ssim_outside(ref, photo_img, outside)
        shift, dx, dy = best_shift(ref, photo_img, outside, a.max_shift)

        rb = cv2.GaussianBlur(ref, (0, 0), a.blur)
        ib = cv2.GaussianBlur(img, (0, 0), a.blur)
        blur = ssim_outside(rb, ib, outside)
        all_, adx, ady = best_shift(rb, photometric_match(rb, ib, outside), outside, a.max_shift)

        rows.append(dict(image=Path(r.path).name, raw=raw, photo=photo, shift=shift, blur=blur,
                         all=all_, dx=dx, dy=dy, all_dx=adx, all_dy=ady))
        print(f"  {k}/{len(pairs)} {Path(r.path).name}: raw {raw:.3f} photo {photo:.3f} "
              f"shift {shift:.3f} ({dx:+d},{dy:+d}) blur {blur:.3f} all {all_:.3f}")

    res = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    print(f"\nMedian over {len(res)} images ({a.condition}, outside glare):")
    print(res[["raw", "photo", "shift", "blur", "all"]].median().round(3).to_string())
    print(f"\nbest shift on top of photo:  median |dx| {res.dx.abs().median():.0f}, "
          f"median |dy| {res.dy.abs().median():.0f}, "
          f"{(res[['dx', 'dy']].abs().max(axis=1) > 0).mean() * 100:.0f}% of images not at (0,0)")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
