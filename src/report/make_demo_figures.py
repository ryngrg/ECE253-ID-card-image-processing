#!/usr/bin/env python3
"""
Whole-card demonstration figures (results/figures/demo_*.jpg; not in the paper or slides, which use make_figures.py).

  demo_ours.jpg        one of our glare photos: reference | untreated | core + halo overlay |
                       classical | DL zero-shot | DL fine-tuned
  demo_midv_photo.jpg  the same panels on a MIDV-2020 highlight photo (reference = the scan)
  demo_midv_clip.jpg   multi-frame fusion on a MIDV-2020 clip: scan | key frame | key frame glare |
                       another frame whose glare is elsewhere | best frame per field | fused

Zones in the overlay are the scoring zones (box_ocr.py): core = classical_glare.detect_core of the
untreated image, halo = glare region (evaluate.glare_region for our cards, the precomputed tone-matched
scan mask for MIDV) minus the core.  Halo specks < MIN_SPECK px are not drawn (readability only).
The clip panels use fuse_frames.glare_stack, i.e. exactly the glare test fusion uses.

Usage
  python src/report/make_demo_figures.py [--ours F03_S_G_1] [--midv fin_id_68] [--clip fin_id_95_mid]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from glare.classical_glare import DEFAULTS, detect_core
from common.evaluate import glare_region
from glare.fuse_frames import glare_stack
from glare.prepare_midv_clips import load_stack

CORE = (40, 110, 235)      # RGB
HALO = (255, 140, 0)
OTHER = (230, 40, 120)     # second frame's glare in the clip figure
BG = (246, 244, 239)
INK = (20, 33, 61)
MIN_SPECK = 500
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def drop_specks(m, min_px=MIN_SPECK):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px])


def outline(img, m, col, t=3):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, col, t)


def zone_overlay(bgr, core, halo):
    """Halo: light orange tint + outline; core: blue fill + outline (RGB out)."""
    out = rgb(bgr).astype(np.float32)
    out[halo] = 0.8 * out[halo] + 0.2 * np.array(HALO)
    out[core] = 0.45 * out[core] + 0.55 * np.array(CORE)
    out = out.clip(0, 255).astype(np.uint8)
    outline(out, halo | core, HALO)
    outline(out, core, CORE)
    return out


def grid(panels, cols, title, legend=None, pad=20, head=56, out=None):
    """panels: list of (RGB image, label); all resized to the first panel's size."""
    h, w = panels[0][0].shape[:2]
    rows = -(-len(panels) // cols)
    top = 70
    leg_h = 60 * len(legend) + 10 if legend else 0
    W = cols * w + (cols + 1) * pad
    H = top + rows * (h + head) + rows * pad + leg_h
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((pad, 14), title, fill=INK, font=font(44))
    for i, (img, lab) in enumerate(panels):
        r, c = divmod(i, cols)
        x, y = pad + c * (w + pad), top + r * (h + head + pad)
        d.text((x, y + 8), lab, fill=INK, font=font(36))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(img), (x, y + head))
    if legend:
        y = H - leg_h + 5
        for items in legend:
            x = pad
            for col, txt in items:
                if col is not None:
                    d.rectangle([x, y, x + 40, y + 40], fill=col)
                    x += 54
                d.text((x, y + 4), txt, fill=INK, font=font(32))
                x += int(d.textlength(txt, font=font(32))) + 50
            y += 60
    canvas.save(out, quality=90)
    print(f"wrote {out}  ({W}x{H})")


ZONE_LEGEND = [[(CORE, "core: clipped (all channels ≥ 248), text gone"),
                (HALO, "halo: > 40 grey levels brighter than the reference, not clipped")]]


def single_image_figure(name, img, ref, glare, out_dirs, title, ref_label, out):
    core = detect_core(img, DEFAULTS)
    halo = drop_specks(glare & ~core)
    panels = [(rgb(ref), ref_label), (rgb(img), "untreated"),
              (zone_overlay(img, core, halo), "glare zones (core / halo)")]
    for lab, d in zip(("classical", "DL zero-shot", "DL fine-tuned"), out_dirs):
        panels.append((rgb(cv2.imread(str(d / name))), lab))
    grid(panels, 3, title, ZONE_LEGEND, out=out)


def clip_figure(a, out):
    pairs = pd.read_csv("out/pairs_midv_clips.csv")
    row = pairs[pairs.path.str.endswith(f"/{a.clip}.png")].iloc[0]
    frames = pd.read_csv("out/midv_clips_frames.csv")
    frames = frames[frames.card_id == row.card_id].copy()
    key = cv2.imread(row.path)
    h, w = key.shape[:2]
    idx, stack, clipped, valid = load_stack(frames, Path(a.root), row.key_idx, (w, h))
    kp = int(np.flatnonzero(idx == row.key_idx)[0])
    G = glare_stack(stack, clipped, valid)
    gk = G[kp] & valid[kp]
    free = ~G
    free[kp] = False
    count = free.sum(axis=0)

    # another frame: covers most of the key frame's glare while having visible glare of its own
    own = np.array([(G[k] & valid[k]).mean() for k in range(len(idx))])
    cover = np.array([free[k][gk].mean() if gk.any() else 0 for k in range(len(idx))])
    cand = [k for k in range(len(idx)) if k != kp and own[k] >= 0.5 * own[kp]] or [k for k in range(len(idx)) if k != kp]
    ko = max(cand, key=lambda k: cover[k])

    ref = cv2.imread(row.ref_path)
    name = f"{a.clip}.png"
    fused = cv2.imread(f"out/midv_clips/fused/{name}")
    best_field = cv2.imread(f"out/midv_clips/best_field/{name}")

    def tint(bgr, m, col, a_=0.45):
        o = rgb(bgr).astype(np.float32)
        o[m] = (1 - a_) * o[m] + a_ * np.array(col)
        o = o.clip(0, 255).astype(np.uint8)
        outline(o, m, col)
        return o

    other = tint(stack[ko], G[ko] & valid[ko], OTHER)
    outline(other, gk, HALO, 3)  # where the key frame's glare is

    n_rec = (count[gk] > 0).mean() if gk.any() else np.nan
    panels = [(rgb(ref), "scan (reference)"),
              (rgb(key), f"key frame = anchor (frame {row.key_idx})"),
              (tint(key, gk, HALO, 0.35), f"key frame glare ({gk.mean():.0%} of document)"),
              (other, f"another aligned frame (frame {idx[ko]})"),
              (rgb(best_field), "best frame per field"),
              (rgb(fused), "fused (only glare pixels replaced)")]
    legend = [[(HALO, "key frame glare"), (OTHER, "other frame's glare: elsewhere"),
               (None, f"{n_rec:.0%} of key-frame glare pixels are clean in ≥ 1 other frame")]]
    grid(panels, 3, f"Multi-frame fusion, MIDV-2020 clip {row.card_id} ({len(idx)} aligned frames)", legend, out=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="F03_S_G_1")
    ap.add_argument("--midv", default="fin_id_68")
    ap.add_argument("--clip", default="fin_id_95_mid")
    ap.add_argument("--root", default="data/external/midv2020")
    ap.add_argument("--out-dir", default="results/figures", type=Path)
    ap.add_argument("--only", default="ours,midv,clip")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    only = a.only.split(",")

    if "ours" in only:
        pairs = pd.read_csv("out/pairs.csv")
        r = pairs[pairs.path.str.contains(f"/{a.ours}_rectified")].iloc[0]
        img, ref = cv2.imread(r.path), cv2.imread(r.ref_path)
        glare = glare_region(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        single_image_figure(Path(r.path).name, img, ref, glare,
                            [Path("out/classical"), Path("out/dl_zeroshot"), Path("out/dl_ft")],
                            f"Our card {a.ours} ({r.split} split)", "reference (same card, diffuse light)",
                            a.out_dir / "demo_ours.jpg")
    if "midv" in only:
        pairs = pd.read_csv("out/pairs_midv.csv")
        r = pairs[pairs.card_id == a.midv].iloc[0]
        img, ref = cv2.imread(r.path), cv2.imread(r.ref_path)
        glare = cv2.imread(r.glare_mask_path, cv2.IMREAD_GRAYSCALE) > 0
        single_image_figure(Path(r.path).name, img, ref, glare,
                            [Path("out/midv/classical"), Path("out/midv/dl_zeroshot"), Path("out/midv/dl_ft")],
                            f"MIDV-2020 photo {a.midv} (external test, {r.device})", "reference (flatbed scan)",
                            a.out_dir / "demo_midv_photo.jpg")
    if "clip" in only:
        clip_figure(a, a.out_dir / "demo_midv_clip.jpg")


if __name__ == "__main__":
    main()
