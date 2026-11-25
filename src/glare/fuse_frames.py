#!/usr/bin/env python3
"""
Multi-frame glare fusion on the MIDV-2020 highlight clips (README "Multi-frame fusion"), and its baselines.

Glare moves with the camera, the document does not: after alignment (prepare_midv_clips.py), a pixel
that is washed out in the key frame is often clean in other frames.  Nothing here is learned or tuned
on the clips; every constant below was fixed before the first evaluation.

Per clip and key-frame rule (keymid / keymax), from the aligned stack in the key frame's geometry and
exposure (prepare_midv_clips.load_stack):
  glare test   a pixel of frame i is "glare" if it is clipped (min channel >= 248) or its grey level
               (Gaussian sigma SMOOTH) is > DELTA above the glare-free level of that pixel = the Q_REF-th
               percentile over all frames; grown by GROW px.  The plan said "temporal median": glare only
               adds light and often covers the same spot in more than half of the frames, where the
               median is glare itself; a low percentile (not the minimum: noise, sub-pixel misalignment
               at stroke edges) is the glare-free level.
Methods (outputs keep the key frame's file name, so box_ocr.py scores them like any other run)
  fused        main method: ONLY the key frame's glare pixels are replaced, by the mean of that pixel
               over the frames where it is not glare (feathered edge, sigma FEATHER); pixels that are
               glare in every frame, and all other key-frame pixels, are left as they are, so residual
               misalignment cannot blur clean text
  median       ablation: plain per-channel temporal median of all frames (no glare test)
  best_frame   baseline: the single frame with the least glare (same test) inside the labelled field
               boxes (no reference used); ties -> closest in time to the key frame
  best_field   strong baseline ("automatic retake per field"): each labelled field box (+ BOX_PAD px) is
               copied from the frame where it has the least glare; ties -> the key frame / closest to it
Diagnostics (results/glare/midv_clips_fusion.csv, one row per clip and rule)
  recoverable_core   share of the key frame's clipped-core pixels (classical_glare.detect_core, the
                     core zone of box_ocr.py) that are outside the core in >= 1 other aligned frame:
                     how often recovery is physically possible
  recoverable_glare  share of the key frame's glare pixels (glare test) with >= 1 glare-free frame
  replaced_frac      share of the document replaced in `fused`
  best_frame_idx, best_field_switched (fields taken from another frame than the key)

Usage
  python src/glare/fuse_frames.py --jobs 3                         # all clips, both rules (~15 min)
  python src/glare/fuse_frames.py --figure                          # + results/figures/midv_clips_fusion.jpg
  then box_ocr.py / compare_runs.py, see run_glare_experiments.sh step "clips"
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from glare.classical_glare import DEFAULTS, detect_core
from glare.prepare_midv_clips import box_mask, load_stack

DELTA = 30.0      # grey levels above the glare-free level
Q_REF = 25        # percentile over time = glare-free level
SMOOTH = 2.0      # px, grey blur for the glare test (ignores single-pixel noise / edge jitter)
GROW = 5          # px, dilation of each frame's glare mask (soft glare edge)
FEATHER = 2.0     # px, sigma of the blend edge in `fused`
BOX_PAD = 4       # px around each field box in best_field (box_ocr crops with pad 2)
METHODS = ("fused", "median", "best_frame", "best_field")
CHUNK = 64        # rows per chunk for percentiles / medians (memory)


def masked_percentile(x, valid, q):
    """Per-pixel q-th percentile over axis 0 of x [N, H, W(, C)] using only the valid frames
    (valid [N, H, W]; nearest rank), in row chunks.  NaN where no frame is valid."""
    out = np.empty(x.shape[1:], np.float32)
    for y in range(0, x.shape[1], CHUNK):
        c = x[:, y:y + CHUNK].astype(np.float32)
        v = valid[:, y:y + CHUNK]
        vb = v[..., None] if c.ndim == 4 else v
        c = np.where(vb, c, np.inf)
        c.sort(axis=0)
        nv = vb.sum(axis=0)
        k = np.clip(np.floor(q / 100 * (nv - 1) + 0.5), 0, None).astype(np.int64)
        r = np.take_along_axis(c, k[None], axis=0)[0]
        out[y:y + CHUNK] = np.where(nv > 0, r, np.nan)
    return out


def glare_stack(stack, clipped, valid):
    """Per-frame glare masks [N, H, W] (glare test above); pixels outside the image count as glare."""
    n = len(stack)
    gray = np.empty(stack.shape[:3], np.float32)
    for k in range(n):
        gray[k] = cv2.GaussianBlur(cv2.cvtColor(stack[k], cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), SMOOTH)
    ref = masked_percentile(gray, valid, Q_REF)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1,) * 2)
    G = np.empty(stack.shape[:3], bool)
    for k in range(n):
        g = (clipped[k] | (gray[k] - ref > DELTA)).astype(np.uint8)
        G[k] = (cv2.dilate(g, ker) > 0) | ~valid[k]
    return G


def closest(cands, idx, key_idx):
    """Among positions `cands` (into idx), the one closest in time to the key frame."""
    return int(cands[np.argmin(np.abs(idx[cands] - key_idx))])


def fuse_one(row, frames: pd.DataFrame, layout: pd.DataFrame, root: Path, out_root: Path):
    t0 = time.time()
    key = cv2.imread(row.path)
    h, w = key.shape[:2]
    name = Path(row.path).name
    idx, stack, clipped, valid = load_stack(frames, root, row.key_idx, (w, h))
    kp = int(np.flatnonzero(idx == row.key_idx)[0])
    t_load = time.time() - t0
    G = glare_stack(stack, clipped, valid)
    n = len(idx)
    others = np.arange(n) != kp

    # fused: replace the key frame's glare pixels by the mean over glare-free frames
    free = ~G
    free[kp] = False
    count = free.sum(axis=0)
    acc = np.zeros((h, w, 3), np.float32)
    for k in np.flatnonzero(others):
        acc += stack[k] * free[k][..., None]
    mean = acc / np.maximum(count, 1)[..., None]
    R = G[kp] & (count > 0)
    wgt = np.maximum(cv2.GaussianBlur(R.astype(np.float32), (0, 0), FEATHER), R)
    wgt[count == 0] = 0
    fused = np.clip(wgt[..., None] * mean + (1 - wgt[..., None]) * key, 0, 255).round().astype(np.uint8)

    # ablation: temporal median of all frames (inside the image)
    med = masked_percentile(stack, valid, 50)
    med = np.where(np.isnan(med), key, med).round().astype(np.uint8)

    # baselines on the labelled field boxes
    lab = layout[layout.labelled]
    in_boxes = box_mask((h, w), lab, pad=2)
    score = G[:, in_boxes].sum(axis=1)
    bf = closest(np.flatnonzero(score == score.min()), idx, row.key_idx)
    best_field = key.copy()
    switched = 0
    for b in lab.itertuples():
        x0, y0 = max(int(round(b.x0)) - BOX_PAD, 0), max(int(round(b.y0)) - BOX_PAD, 0)
        x1, y1 = min(int(round(b.x1)) + BOX_PAD, w), min(int(round(b.y1)) + BOX_PAD, h)
        if x1 <= x0 or y1 <= y0:
            continue
        frac = G[:, y0:y1, x0:x1].mean(axis=(1, 2))
        f = closest(np.flatnonzero(frac == frac.min()), idx, row.key_idx)
        if f != kp:
            best_field[y0:y1, x0:x1] = stack[f][y0:y1, x0:x1]
            switched += 1

    for m, img in zip(METHODS, (fused, med, stack[bf], best_field)):
        cv2.imwrite(str(out_root / m / name), img)

    # diagnostics
    core_k = detect_core(key, DEFAULTS)
    if core_k.any():
        ok = np.zeros((h, w), bool)
        for k in np.flatnonzero(others):
            ok |= ~detect_core(stack[k], DEFAULTS) & valid[k]
        rec_core = float(ok[core_k].mean())
    else:
        rec_core = np.nan
    res = dict(card_id=row.card_id, doc_type=row.doc_type, split=row.split, image=name, n_frames=n,
               key_idx=row.key_idx, key_core_px=int(core_k.sum()), key_core_px_boxes=int((core_k & in_boxes).sum()),
               key_glare_frac=float(G[kp].mean()), recoverable_core=rec_core,
               recoverable_glare=float((count[G[kp]] > 0).mean()) if G[kp].any() else np.nan,
               replaced_frac=float(R.mean()), best_frame_idx=int(idx[bf]), best_frame_is_key=bf == kp,
               best_field_switched=switched, n_fields=len(lab), t_load=t_load, t_fuse=time.time() - t0 - t_load)
    print(f"{name:34s} {n:3d} frames  key glare {res['key_glare_frac']:.3f}  replaced {res['replaced_frac']:.3f}  "
          f"recoverable core {rec_core:.2f}  best frame {idx[bf]}  fields switched {switched}/{len(lab)}  "
          f"({t_load:.0f}+{res['t_fuse']:.0f}s)", flush=True)
    return res


def work(args):
    cv2.setNumThreads(2)
    return fuse_one(*args)


def figure(pairs, fus, out_root: Path, path: Path, n=4):
    """key frame | fused | best single frame | best frame per field | scan, for the n keymax clips with the
    most clipped core inside the field boxes."""
    sel = fus[fus.split == "keymax"].nlargest(n, "key_core_px_boxes")
    rows = []
    for r in sel.itertuples():
        p = pairs[pairs.path.map(lambda s: Path(s).name) == r.image].iloc[0]
        ims = [cv2.imread(p.path)] + [cv2.imread(str(out_root / m / r.image)) for m in ("fused", "best_frame", "best_field")]
        ims.append(cv2.imread(p.ref_path))
        hh = 300
        ims = [cv2.resize(x, (round(x.shape[1] * hh / x.shape[0]), hh), interpolation=cv2.INTER_AREA) for x in ims]
        for x, t in zip(ims, ("key frame (untreated)", "fused", "best single frame", "best frame per field", "scan")):
            cv2.putText(x, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(x, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(ims))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/midv2020", type=Path)
    ap.add_argument("--pairs", default="out/pairs_midv_clips.csv", type=Path)
    ap.add_argument("--frames", default="out/midv_clips_frames.csv", type=Path)
    ap.add_argument("--layout", default="results/layouts/box_layout_midv_clips.csv", type=Path)
    ap.add_argument("--out-root", default="out/midv_clips", type=Path)
    ap.add_argument("--results", default="results/glare/midv_clips_fusion.csv", type=Path)
    ap.add_argument("--splits", default="keymid,keymax")
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--figure", action="store_true", help="also write results/figures/midv_clips_fusion.jpg")
    ap.add_argument("--figure-only", action="store_true")
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs.usable & pairs.split.isin(a.splits.split(","))]
    if a.only:
        pairs = pairs[pairs.card_id.isin(a.only.split(","))]
    if not a.figure_only:
        frames = pd.read_csv(a.frames, keep_default_na=False, na_values=[""])
        frames["aligned"] = frames.aligned.astype(str) == "True"
        layout = pd.read_csv(a.layout, keep_default_na=False)
        layout["labelled"] = layout.labelled.astype(str) == "True"
        for m in METHODS:
            (a.out_root / m).mkdir(parents=True, exist_ok=True)
        jobs = [(r, frames[frames.card_id == r.card_id], layout[layout.card_id == r.card_id], a.root, a.out_root)
                for _, r in pairs.iterrows()]
        with Pool(a.jobs) as pool:
            res = pool.map(work, jobs, chunksize=1) if a.jobs > 1 else list(map(work, jobs))
        fus = pd.DataFrame(res)
        if a.only and a.results.exists():
            old = pd.read_csv(a.results)
            fus = pd.concat([old[~old.image.isin(fus.image)], fus])
        fus.to_csv(a.results, index=False)
        for s, g in fus.groupby("split"):
            print(f"{s}: {len(g)} clips, median frames {g.n_frames.median():.0f}; key frames with a clipped core: "
                  f"{(g.key_core_px > 0).sum()} (in a field box: {(g.key_core_px_boxes > 0).sum()}); "
                  f"recoverable core median {g.recoverable_core.median():.2f}, "
                  f"recoverable glare median {g.recoverable_glare.median():.2f}; "
                  f"replaced median {g.replaced_frac.median():.3f}; best frame = key in {g.best_frame_is_key.sum()}; "
                  f"fields switched {g.best_field_switched.sum()}/{g.n_fields.sum()}")
        print(f"Saved {a.results}, {a.out_root}/{{{','.join(METHODS)}}}/")
    if a.figure or a.figure_only:
        figure(pairs, pd.read_csv(a.results), a.out_root, Path("results/figures/midv_clips_fusion.jpg"))


if __name__ == "__main__":
    main()
