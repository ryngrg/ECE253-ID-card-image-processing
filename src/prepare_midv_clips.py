#!/usr/bin/env python3
"""
MIDV-2020 "highlight present" video clips (90-99 of every type): aligned frame stacks for
multi-frame fusion (NEXT_STEPS step 5; fusion itself is in fuse_frames.py).

Data (see NEXT_STEPS step 5 for the download): data/external/midv2020/clips/images/<type>/<nn>/*.jpg
(every 6th frame of a 4K clip, annotated with the document quad in clips/annotations/<type>/<nn>.json),
plus the scan and the template of the same document (as for the photos, prepare_midv.py).
100 clips of 100 documents, none of which is a photo document; nothing is tuned on them.

Per clip
  1. every annotated frame is warped from its quad to the canonical frame (prepare_midv.canonical_size);
     dropped if < MIN_INSIDE of the document is inside the image ("outside"; 14% of the frames have a
     quad corner outside the image, by a median of 51 px: one edge cut off, the fields mostly visible) or
     if it is motion-blurred ("blur"):
     variance of the Laplacian (on non-clipped pixels) < BLUR_REL x the clip's median
     (fixed in advance: blur in these clips is bimodal, ~0.1 vs ~1 x median).
     Pixels outside the image are tracked per frame (in_frame mask) and treated as missing everywhere
     (alignment, gain, fusion); key frames (both rules) must show the whole document;
  2. anchor = the middle frame of the kept ones with the whole document visible = key frame of rule "mid".  It is aligned to the scan
     with prepare_midv.safe_ecc, exactly like the photos (quad-only if the guard rejects it), so the
     template boxes and the scan reference apply;
  3. every other kept frame is aligned to the anchor by a homography: SIFT + RANSAC initial guess, refined
     by ECC on locally normalised images (Gaussian high-pass divided by the local std, sigma 6 px: removes
     the glare veil, which otherwise dominates the correlation) with clipped pixels of BOTH frames set to 0
     (class Aligner).  No non-rigid refinement: on held-out quadrants, a polynomial displacement field
     fitted to local shifts extrapolated worse than the homography alone (even on a bent passport page),
     and fusion replaces pixels exactly where nothing can be measured (under the glare).  Guard: the
     correlation must rise by > 0.005 and no corner may move > MAX_SHIFT px (safe_ecc uses 15 px; annotated
     quads with a corner outside the image are 20-60 px off), plus a reference-free check of the
     result: median phase-correlation shift over >= MIN_TILES clean 96 px tiles <= MAX_RESID px
     ("ecc" / "resid" drops).  Typical residual of accepted frames: ~0.3 px;
  4. per-frame colour gain to the anchor (per channel median ratio over pixels unclipped in both)
     against auto-exposure changes;
  5. key frame of rule "max" = the aligned whole-document frame with the most clipped core (classical_glare.detect_core,
     the same core as the zones of box_ocr.py) inside the labelled field boxes; ties -> closest to the anchor.
Each kept frame is stored as ONE homography from the raw 4K frame to the anchor geometry
(quad warp composed with the ECC warp), so a frame is interpolated once; load_stack() re-warps them.

Both key-frame rules are written as separate "splits" of one pairs file (keymid / keymax), with
image names {type}_{nn}_{rule}.png, so box_ocr.py, compare_runs.py and the method scripts work unchanged.

Outputs
  data/midv_clips/rectified/{doc}_{nn}_{mid|max}.png   key frame in the scan frame       ("untreated")
  data/midv_clips/reference/{doc}_{nn}.png             rectified scan                     (ceiling run)
  data/midv_clips/glare_masks/{doc}_{nn}_{rule}.png    glare region of the key frame (prepare_midv rule)
  data/midv_clips/overlays/{doc}_{nn}.jpg              key mid | key max | glare + boxes  (visual check)
  out/pairs_midv_clips.csv                             one row per (clip, rule): split = keymid / keymax
  out/midv_clips_frames.csv                            one row per annotated frame: drop reason, ECC, residual,
                                                       homography raw -> anchor, gain, clipped px in boxes
  results/box_layout_midv_clips.csv                    fixed boxes + targets (box_ocr.py --layout)

Usage
  python src/prepare_midv_clips.py --jobs 4                 # ~10 min on an M2
  python src/prepare_midv_clips.py --only alb_id_90,esp_id_93
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classical_glare import DEFAULTS, detect_core
from prepare_midv import (TYPES, canonical_size, doc_quad, glare_mask, layout_rows, match_tone, regions,
                          safe_ecc, template_boxes, warp)

NUMS = range(90, 100)    # "highlight present" clips
BLUR_REL = 0.5           # drop frames with variance of Laplacian < BLUR_REL x clip median
MAX_SHIFT = 80.0         # px, max corner move vs the quad warp (quads with a corner outside the image
                         # are off by 20-60 px; wrong alignments move corners by 150-800 px)
MAX_RESID = 1.0          # px, median tile residual after alignment
MIN_TILES = 5            # clean tiles needed to check the residual
TILE = 96
NORM_SIGMA = 6.0         # px, local normalisation for the frame-to-frame ECC
RULES = ("mid", "max")
MIN_INSIDE = 0.8         # min fraction of the document inside the image


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def clip_frames(root: Path, doc: str, n: int):
    """[(filename, quad)] of the annotated frames, in time order."""
    meta = json.loads((root / "clips" / "annotations" / doc / f"{n:02d}.json").read_text())["_via_img_metadata"]
    return [(v["filename"], doc_quad(v["regions"])) for v in sorted(meta.values(), key=lambda v: v["filename"])]


def frame_path(root: Path, doc: str, n: int, fname: str) -> Path:
    return root / "clips" / "images" / doc / f"{n:02d}" / fname


def quad_homography(quad, size):
    w, h = size
    return cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))


def warp_raw(raw, M, size):
    """Output pixel x = raw(M x): M maps canonical coordinates to raw-frame coordinates."""
    return cv2.warpPerspective(raw, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def in_frame(raw_shape, M, size, erode=3):
    """Pixels of the output (x = raw(M x)) whose source lies inside the raw frame."""
    ones = np.ones(raw_shape[:2], np.uint8)
    v = cv2.warpPerspective(ones, M, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.erode(v, np.ones((2 * erode + 1,) * 2, np.uint8)) > 0


def clipped_mask(img, valid=None, thr=235, grow=15):
    """Clipped pixels (grown), plus the pixels outside the image (grown) if a valid mask is given."""
    bad = ((img.min(axis=2) >= thr) * 255).astype(np.uint8)
    if valid is not None:
        bad[~valid] = 255
    return cv2.dilate(bad, np.ones((grow, grow), np.uint8)) > 0


def sharpness(img, valid=None) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok = ~clipped_mask(img, valid)
    return float(cv2.Laplacian(g, cv2.CV_64F)[ok].var()) if ok.sum() > 1000 else 0.0


def structure(img, valid=None, s=NORM_SIGMA):
    """Locally normalised grey image (veil-free), clipped / out-of-image pixels set to 0; and the valid mask."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = g - cv2.GaussianBlur(g, (0, 0), s)
    z = np.clip(d / (np.sqrt(cv2.GaussianBlur(d * d, (0, 0), s)) + 4), -3, 3)
    bad = clipped_mask(img, valid)
    z[bad] = 0
    return z, (~bad * 255).astype(np.uint8)


class Aligner:
    """Homography of a frame onto the anchor: SIFT + RANSAC on the grey images for the initial guess,
    then ECC at full resolution on the normalised images (glare masked in both) to refine it.
    (Pyramid ECC from the identity failed on ~25% of the sharp frames of some clips; SIFT + RANSAC
    alone leaves ~0.3 px.)  If SIFT finds < MIN_INLIERS inliers, ECC starts from the identity
    (pyramid); if ECC fails, the SIFT homography is kept and judged by the same guard."""
    MIN_INLIERS = 15

    def __init__(self, ref, valid=None):
        self.sift = cv2.SIFT_create(4000)
        self.zr, mr = structure(ref, valid)
        self.kr, self.dr = self.sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), mr)
        self.matcher = cv2.BFMatcher()

    def sift_init(self, img, mask):
        k, d = self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
        if d is None or len(k) < self.MIN_INLIERS or self.dr is None:
            return None, 0
        good = [m[0] for m in self.matcher.knnMatch(self.dr, d, k=2)
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < self.MIN_INLIERS:
            return None, 0
        Hs, inl = cv2.findHomography(np.float32([self.kr[m.queryIdx].pt for m in good]),
                                     np.float32([k[m.trainIdx].pt for m in good]), cv2.RANSAC, 3.0)
        n_inl = int(inl.sum()) if inl is not None else 0
        return (Hs if n_inl >= self.MIN_INLIERS else None), n_inl

    def __call__(self, img, valid=None):
        """Returns (W, cc0, cc1, n_inliers, method): img(W x) ~ ref(x); W = None if all failed.
        cc0 = ECC correlation of the quad-only warp, cc1 = of the final warp (normalised images)."""
        zi, mi = structure(img, valid)
        h, w = zi.shape
        cc0 = float(cv2.computeECC(self.zr, zi, mi))
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        Hs, n_inl = self.sift_init(img, mi)
        if Hs is not None:
            try:
                cc, We = cv2.findTransformECC(self.zr, zi, Hs.astype(np.float32), cv2.MOTION_HOMOGRAPHY, crit, mi, 5)
                return We.astype(np.float64), cc0, float(cc), n_inl, "sift+ecc"
            except cv2.error:
                zw = cv2.warpPerspective(zi, Hs, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                mw = cv2.warpPerspective(mi, Hs, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                return Hs, cc0, float(cv2.computeECC(self.zr, zw, mw)), n_inl, "sift"
        W = np.eye(3)                       # no SIFT guess: coarse-to-fine ECC from the quad warp
        for lev in (2, 1, 0):
            s = 0.5 ** lev
            S = np.diag([s, s, 1.0])
            if lev:
                r, i = (cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for x in (self.zr, zi))
                m = cv2.resize(mi, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            else:
                r, i, m = self.zr, zi, mi
            try:
                cc, Ws = cv2.findTransformECC(r, i, (S @ W @ np.linalg.inv(S)).astype(np.float32),
                                              cv2.MOTION_HOMOGRAPHY, crit, m, 5)
            except cv2.error:
                return None, cc0, np.nan, n_inl, "failed"
            W = np.linalg.inv(S) @ Ws.astype(np.float64) @ S
        return W, cc0, float(cc), n_inl, "ecc"


def tile_residual(ref, al, vr=None, va=None):
    """Median local shift (px) between two aligned images: phase correlation of the normalised
    images on TILE x TILE tiles that are clean (no clipped / out-of-image pixel) and textured in both."""
    zr, mr = structure(ref, vr)
    za, ma = structure(al, va)
    h, w = zr.shape
    win = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)
    out = []
    for y in range(0, h - TILE + 1, TILE):
        for x in range(0, w - TILE + 1, TILE):
            sl = np.s_[y:y + TILE, x:x + TILE]
            if mr[sl].min() == 0 or ma[sl].min() == 0 or zr[sl].std() < 0.5 or za[sl].std() < 0.5:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(zr[sl], za[sl], win)
            if resp > 0.1:
                out.append(np.hypot(dx, dy))
    return (float(np.median(out)) if out else np.nan), len(out)


def channel_gain(ref, img, vr=None, va=None):
    """Per-channel g such that ref ~ g * img (median ratio over pixels unclipped, inside the image and
    not dark in both)."""
    ok = ~clipped_mask(ref, vr) & ~clipped_mask(img, va) & (ref.min(axis=2) > 20) & (img.min(axis=2) > 20)
    if ok.sum() < 1000:
        return np.ones(3)
    return np.array([np.median(ref[..., c][ok].astype(np.float32) / img[..., c][ok]) for c in range(3)])


def box_mask(shape, boxes: pd.DataFrame, pad=0):
    m = np.zeros(shape, bool)
    h, w = shape
    for x0, y0, x1, y1 in boxes[["x0", "y0", "x1", "y1"]].to_numpy():
        m[max(int(round(y0)) - pad, 0):min(int(round(y1)) + pad, h),
          max(int(round(x0)) - pad, 0):min(int(round(x1)) + pad, w)] = True
    return m


# --------------------------------------------------------------------------
# Stack loading (used by fuse_frames.py)
# --------------------------------------------------------------------------
def load_stack(frames: pd.DataFrame, root: Path, key_idx: int, size):
    """Aligned frames of one clip (rows of out/midv_clips_frames.csv with aligned == True), re-warped
    from the raw 4K JPEGs and colour-corrected to the key frame's exposure.
    Returns (idx [N], stack uint8 [N,H,W,3], clipped bool [N,H,W] (min channel >= 248 before the gain),
    valid bool [N,H,W] (source pixel inside the raw frame))."""
    fr = frames[frames.aligned].sort_values("idx")
    w, h = size
    stack = np.empty((len(fr), h, w, 3), np.uint8)
    clipped = np.empty((len(fr), h, w), bool)
    valid = np.empty((len(fr), h, w), bool)
    gk = fr.loc[fr.idx == key_idx, ["gain_b", "gain_g", "gain_r"]].to_numpy()[0]
    for k, r in enumerate(fr.itertuples()):
        raw = cv2.imread(str(frame_path(root, r.doc_type, r.num, r.frame)))
        M = np.array(json.loads(r.M))
        al = warp_raw(raw, M, size)
        valid[k] = in_frame(raw.shape, M, size)
        clipped[k] = (al.min(axis=2) >= DEFAULTS["clip_thresh"]) & valid[k]
        g = np.array([r.gain_b, r.gain_g, r.gain_r]) / gk
        stack[k] = np.clip(al.astype(np.float32) * g.astype(np.float32), 0, 255).round().astype(np.uint8)
    return fr.idx.to_numpy(), stack, clipped, valid


# --------------------------------------------------------------------------
# One clip
# --------------------------------------------------------------------------
def process_clip(args):
    root, out, doc, n = args
    cv2.setNumThreads(2)
    t0 = time.time()
    cid = f"{doc}_{n:02d}"
    tmpl = cv2.imread(str(root / "templates" / "images" / doc / f"{n:02d}.jpg"))
    size = W, H = canonical_size(doc, tmpl)
    scan = warp(cv2.imread(str(root / "scan_upright" / "images" / doc / f"{n:02d}.jpg")),
                doc_quad(regions(root, "scan_upright", doc, n)), size)
    boxes, ok_t = template_boxes(root, doc, n, tmpl, scan)
    lay = pd.DataFrame(layout_rows(cid, doc, boxes))
    in_boxes = box_mask((H, W), lay[lay.labelled])

    # 1. quad warp, outside / blur
    rows, qw = [], {}
    for idx, (fname, quad) in enumerate(clip_frames(root, doc, n)):
        raw = cv2.imread(str(frame_path(root, doc, n, fname)))
        Hq = quad_homography(quad, size)
        vq = in_frame(raw.shape, np.linalg.inv(Hq), size, erode=0)
        row = dict(card_id=cid, doc_type=doc, num=n, idx=idx, frame=fname, inside_frac=float(vq.mean()),
                   whole_doc=bool(vq.all()), doc_width_px=float(np.linalg.norm(quad[1] - quad[0])),
                   sharpness=np.nan, drop_reason="")
        if row["inside_frac"] >= MIN_INSIDE:
            vq = in_frame(raw.shape, np.linalg.inv(Hq), size)
            qw[idx] = (Hq, warp(raw, quad, size), vq)
            row["sharpness"] = sharpness(qw[idx][1], vq)
        else:
            row["drop_reason"] = "outside"
        rows.append(row)
    F = pd.DataFrame(rows)
    med = F.sharpness.median()
    F["sharp_rel"] = F.sharpness / med
    F.loc[(F.drop_reason == "") & (F.sharp_rel < BLUR_REL), "drop_reason"] = "blur"
    cand = F.idx[F.drop_reason == ""].to_list()
    if not cand:
        print(f"{cid}: no usable frame", flush=True)
        return [], lay.to_dict("records"), F.to_dict("records")

    # 2. anchor -> scan (as the photos)
    whole = [i for i in cand if F.whole_doc[i]] or cand
    anchor = whole[len(whole) // 2]
    Hq_a, aq, _ = qw[anchor]
    Wp, scc0, scc1, ok_p = safe_ecc(scan, aq)

    def aligned(idx, We):
        """Homography raw -> anchor geometry (quad warp, then We), the frame warped once from raw,
        and its in-frame mask."""
        M = np.linalg.inv(qw[idx][0]) @ We
        raw = cv2.imread(str(frame_path(root, doc, n, F.frame[idx])))
        return M, warp_raw(raw, M, size), in_frame(raw.shape, M, size)

    M_a, A, VA = aligned(anchor, Wp)
    corners = np.float32([[[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]])
    for c in ("ecc_cc0", "ecc_cc1", "ecc_shift", "resid", "n_tiles", "gain_b", "gain_g", "gain_r", "core_px_boxes",
              "clipped_px"):
        F[c] = np.nan
    F["M"] = ""
    F["n_inliers"] = np.nan
    F["align_method"] = ""
    align = Aligner(A, VA)
    for idx in cand:
        if idx == anchor:
            M, al, va, cc0, cc1, shift, res, nt = M_a, A, VA, 1.0, 1.0, 0.0, 0.0, -1
        else:
            We, cc0, cc1, n_inl, meth = align(qw[idx][1], qw[idx][2])
            shift = (np.abs(cv2.perspectiveTransform(corners, We) - corners).max() if We is not None else np.nan)
            F.loc[idx, ["ecc_cc0", "ecc_cc1", "ecc_shift", "n_inliers"]] = cc0, cc1, shift, n_inl
            F.loc[idx, "align_method"] = meth
            if We is None or not (cc1 > cc0 + 0.005) or shift > MAX_SHIFT:
                F.loc[idx, "drop_reason"] = "ecc"
                continue
            M, al, va = aligned(idx, We)
            res, nt = tile_residual(A, al, VA, va)
            F.loc[idx, ["resid", "n_tiles"]] = res, nt
            if nt < MIN_TILES or not res <= MAX_RESID:
                F.loc[idx, "drop_reason"] = "resid"
                continue
        core = detect_core(al, DEFAULTS) & va
        F.loc[idx, ["ecc_cc0", "ecc_cc1", "ecc_shift", "resid", "n_tiles"]] = cc0, cc1, shift, res, nt
        F.loc[idx, ["gain_b", "gain_g", "gain_r"]] = channel_gain(A, al, VA, va)
        F.loc[idx, "core_px_boxes"] = int((core & in_boxes).sum())
        F.loc[idx, "clipped_px"] = int(((al.min(axis=2) >= DEFAULTS["clip_thresh"]) & va).sum())
        F.loc[idx, "M"] = json.dumps(np.round(M, 10).tolist())
    F["aligned"] = F.M != ""
    al_idx = F.idx[F.aligned].to_numpy()

    # 5. key frames (whole document visible)
    kc = np.array([i for i in al_idx if F.whole_doc[i]] or list(al_idx))
    cp = F.loc[kc, "core_px_boxes"].to_numpy()
    best = kc[cp == cp.max()]
    keymax = int(best[np.argmin(np.abs(best - anchor))])
    keys = {"mid": anchor, "max": keymax}
    F["keymid"] = F.idx == anchor
    F["keymax"] = F.idx == keymax

    # outputs
    ref_path = out / "reference" / f"{cid}.png"
    cv2.imwrite(str(ref_path), scan)
    pairs, vis = [], []
    for rule, k in keys.items():
        key = A if k == anchor else warp_raw(cv2.imread(str(frame_path(root, doc, n, F.frame[k]))),
                                             np.array(json.loads(F.M[k])), size)
        gm = glare_mask(key, match_tone(key, scan))
        name = f"{cid}_{rule}"
        p_path, m_path = out / "rectified" / f"{name}.png", out / "glare_masks" / f"{name}.png"
        cv2.imwrite(str(p_path), key)
        cv2.imwrite(str(m_path), gm.astype(np.uint8) * 255)
        v = key.copy()
        v[gm] = (0.5 * v[gm] + [0, 0, 127]).astype(np.uint8)
        for x in lay.itertuples():
            cv2.rectangle(v, (int(x.x0), int(x.y0)), (int(x.x1), int(x.y1)),
                          (0, 200, 0) if x.labelled else (160, 160, 160), 1)
        vis.append(v)
        pairs.append(dict(card_id=cid, doc_type=doc, num=n, key_rule=rule, key_idx=k, key_frame=F.frame[k],
                          path=str(p_path), ref_path=str(ref_path), glare_mask_path=str(m_path),
                          condition="S_G", split=f"key{rule}",
                          n_annotated=len(F), n_inside=int((F.inside_frac >= MIN_INSIDE).sum()),
                          n_sharp=len(cand), n_aligned=len(al_idx),
                          ecc_used=ok_p, ecc_cc_quad=scc0, ecc_cc=scc1 if ok_p else scc0, template_ecc_used=ok_t,
                          glare_frac=float(gm.mean()), clipped_px=int(F.clipped_px[k]),
                          core_px_boxes=int(F.core_px_boxes[k]), usable=True, problem=""))
    cv2.imwrite(str(out / "overlays" / f"{cid}.jpg"),
                cv2.resize(np.hstack(vis), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    dr = F.drop_reason.value_counts().to_dict()
    print(f"{cid:26s} {len(F):3d} frames -> {len(al_idx):3d} aligned  (dropped {dict(sorted(dr.items()))})  "
          f"median resid {F.resid[F.aligned & (F.idx != anchor)].median():.2f}px  "
          f"scan ecc {'used' if ok_p else 'rejected'}  key mid {anchor} / max {keymax} "
          f"({int(cp.max())} core px in boxes)  {time.time() - t0:.0f}s", flush=True)
    return pairs, lay.to_dict("records"), F.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/midv2020", type=Path)
    ap.add_argument("--out", default="data/midv_clips", type=Path)
    ap.add_argument("--pairs-out", default="out/pairs_midv_clips.csv", type=Path)
    ap.add_argument("--frames-out", default="out/midv_clips_frames.csv", type=Path)
    ap.add_argument("--layout-out", default="results/box_layout_midv_clips.csv", type=Path)
    ap.add_argument("--only", default="", help="comma-separated clip ids, e.g. alb_id_90 (outputs get a _only suffix)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    for d in ("rectified", "reference", "glare_masks", "overlays"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    clips = [(t, n) for t in TYPES for n in NUMS]
    if a.only:
        keep = set(a.only.split(","))
        clips = [c for c in clips if f"{c[0]}_{c[1]:02d}" in keep]
        a.pairs_out, a.frames_out, a.layout_out = (p.with_name(p.stem + "_only" + p.suffix)
                                                   for p in (a.pairs_out, a.frames_out, a.layout_out))
    jobs = [(a.root, a.out, t, n) for t, n in clips]
    with Pool(a.jobs) as pool:
        res = pool.map(process_clip, jobs, chunksize=1) if a.jobs > 1 else list(map(process_clip, jobs))
    P = pd.DataFrame([r for x in res for r in x[0]])
    L = pd.DataFrame([r for x in res for r in x[1]])
    F = pd.DataFrame([r for x in res for r in x[2]])
    for p in (a.pairs_out, a.frames_out, a.layout_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(a.pairs_out, index=False)
    F.to_csv(a.frames_out, index=False)
    L.to_csv(a.layout_out, index=False)
    al = F[F.aligned & ~F.keymid]
    print(f"\n{P.card_id.nunique()} clips; {len(F)} annotated frames, drop reasons "
          f"{F.drop_reason.value_counts().to_dict()}; {F.aligned.sum()} aligned "
          f"(per clip median {F.groupby('card_id').aligned.sum().median():.0f}); "
          f"residual of aligned frames median {al.resid.median():.2f} px, p90 {al.resid.quantile(0.9):.2f} px; "
          f"anchor -> scan ECC used {P[P.key_rule == 'mid'].ecc_used.sum()}/{P.card_id.nunique()}")
    print(f"key max = key mid in {(P[P.key_rule == 'max'].key_idx.to_numpy() == P[P.key_rule == 'mid'].key_idx.to_numpy()).sum()} clips; "
          f"{L.labelled.sum()} labelled boxes of {len(L)}")
    print(f"Saved {a.pairs_out}, {a.frames_out}, {a.layout_out}, {a.out}/")


if __name__ == "__main__":
    main()
