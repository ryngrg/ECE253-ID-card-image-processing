#!/usr/bin/env python3
"""
Batch perspective correction for photos that contain exactly one fully visible
ID card (ID-1 / CR80, 85.6 x 53.98 mm).

Pipeline per image
------------------
1. Downscale to ~1000 px (fast, and suppresses blur/texture).
2. Generate several independent "card vs. background" hypotheses:
     a) edge-based        : blurred Canny -> closed edge map -> contours
     b) colour-based      : Lab distance from the background colour (sampled from
                            the image border), thresholded two ways (Otsu, and a
                            threshold calibrated on the background itself, which
                            catches pale cards that Otsu misses)
     c) seeded GrabCut    : seeded from each colour mask
     d) rect GrabCut      : classic GrabCut initialised with a rectangle just
                            inside the image border (handles pale card on pale
                            table, where colour/edges alone are ambiguous)
3. Turn each hypothesis into a quadrilateral (convex hull -> approxPolyDP).
4. Score every quad (rectangularity, ID-1 aspect ratio, size, not touching the
   image border, hull/quad agreement).  Among near-top scorers pick the one that
   agrees most with the others (consensus), so a single outlier hypothesis
   (e.g. one that follows a plastic sleeve) doesn't win by a hair.
5. Refine: fit a robust line (Huber) to the contour points along each side and
   intersect neighbouring lines.  This recovers the *true* sharp corners even
   though real cards have rounded corners, and it is sub-pixel accurate and
   tolerant of blur.  Corners are scaled back to full resolution.
6. cv2.getPerspectiveTransform to a canonical 1012x638 card (homography H).
7. Reference-based alignment (NEW).  Files named {card}_{B|S}_{G|D}_{num}.png are
   grouped by card; the S_D image is the card's reference and is rectified from
   its corners as above (this defines the canonical frame).  Every other image of
   that card is then registered to the rectified reference with ECC
   (MOTION_HOMOGRAPHY, coarse-to-fine, saturated/glare pixels masked out), giving a
   small correction W.  W is composed with the corner homography,
   H_final = inv(W) @ H, and the RAW photo is warped ONCE with H_final, so there is
   no second interpolation (which would soften the image) and the residual error
   of the corner detection is corrected as a full projective transform, not just
   a shift + rotation.  If ECC fails, correlates poorly, or moves the card by an
   implausible amount, the plain corner-based result is kept and flagged.

Usage
-----
    python rectify_ids.py INPUT_DIR OUTPUT_DIR [--debug] [--width 1012] [--keep-portrait]
                          [--no-align] [--max-shift 30]

--debug also writes <name>_debug.jpg (detected quad drawn on the input) into
OUTPUT_DIR/debug/, keeping the rectified images in OUTPUT_DIR uncluttered.
--no-align skips step 7 (corner-based rectification only).
A per-image summary is written to OUTPUT_DIR/alignment_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]            # H x W x 3 (BGR) or H x W (mask / gray)
Quad = npt.NDArray[np.float32]           # (4, 2) corner array
Contour = npt.NDArray[np.int32]          # OpenCV contour, (N, 1, 2)
Line = tuple[float, float, float, float]  # cv2.fitLine output: (vx, vy, x0, y0)
Scored = tuple[float, Quad, Contour]     # (score, quad, source contour)

ID1_ASPECT = 85.60 / 53.98          # ~1.586
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FILE_RE = re.compile(r"^(?P<card>.+)_(?P<blur>[BS])_(?P<glare>[GD])_(?P<num>\d+)$")  # file stem


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def order_corners(pts: npt.ArrayLike) -> Quad:
    """Return corners as TL, TR, BR, BL (image coords, y down)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]                       # clockwise from "left"
    s = pts.sum(1)
    tl = int(np.argmin(s))
    return np.roll(pts, -tl, axis=0)                 # TL, TR, BR, BL


def intersect(l1: Line, l2: Line) -> npt.NDArray[np.float32] | None:
    """Intersect two lines given as (vx, vy, x0, y0)."""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1], np.float64))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1], np.float32)


def hull_to_quad(hull: Contour) -> Quad:
    """Approximate a convex hull by 4 points; fall back to minAreaRect."""
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.10, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def quad_score(quad: Quad, hull_area: float, shape: tuple[int, ...]) -> float:
    """Higher is better.  Returns -inf for implausible quads."""
    h, w = shape[:2]
    q = order_corners(quad)
    area = cv2.contourArea(q)
    frac = area / (h * w)
    if not (0.04 < frac < 0.95) or area < 1:
        return -np.inf
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return -np.inf

    # interior angles
    angs = []
    for i in range(4):
        a, b, c = q[i - 1] - q[i], q[(i + 1) % 4] - q[i], None
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        angs.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    if min(angs) < 45 or max(angs) > 135:
        return -np.inf

    # aspect ratio (perspective-tolerant)
    wd = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    ht = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
    ar = max(wd, ht) / max(min(wd, ht), 1e-6)
    ar_pen = abs(np.log(ar / ID1_ASPECT))

    # touches border? (the whole card is guaranteed visible)
    m = 0.01 * max(h, w)
    border_pen = float(np.any((q < m) | (q[:, 0] > w - m)[:, None] |
                              (q[:, 1] > h - m)[:, None]))

    fit = min(hull_area, area) / max(hull_area, area)       # 1 == hull is a quad
    angle_pen = np.mean(np.abs(np.array(angs) - 90)) / 45
    return 2.0 * fit - 1.5 * ar_pen - 1.0 * border_pen - 0.7 * angle_pen + 0.5 * np.sqrt(frac)


# --------------------------------------------------------------------------
# segmentation hypotheses -> list of contours
# --------------------------------------------------------------------------
def largest_component_contour(mask: Image) -> list[Contour]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [max(cnts, key=cv2.contourArea)] if cnts else []


def edge_contours(img: Image) -> list[Contour]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    out = []
    for lo_f, hi_f in ((0.5, 1.2), (0.25, 0.7)):
        edges = cv2.Canny(gray, lo_f * med, hi_f * med)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
        out += cnts
    return out


def colour_mask(img: Image) -> tuple[Image, Image]:
    """Foreground = pixels far (in Lab) from the colour distribution of the
    image border.  Works well on textured but fairly uniform surfaces."""
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, int(0.04 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    mu = np.median(border, 0)
    cov = np.cov(border.T) + np.eye(3) * 1e-3
    inv = np.linalg.inv(cov)
    d = lab.reshape(-1, 3) - mu
    dist = np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d)).reshape(h, w)
    dist8 = cv2.normalize(np.clip(dist, 0, np.percentile(dist, 99.5)), None,
                          0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m_otsu = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bd = np.concatenate([dist[:b].ravel(), dist[-b:].ravel(), dist[:, :b].ravel(), dist[:, -b:].ravel()])
    T = max(1.6 * np.percentile(bd, 99), 4.0)
    return m_otsu, ((dist > T) * 255).astype(np.uint8)


def grabcut_mask(img: Image, seed_mask: Image, size: int = 500) -> Image:
    """GrabCut seeded from a rough mask (run at reduced resolution for speed)."""
    H, W = img.shape[:2]
    k = min(1.0, size / max(H, W))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        seed_mask = cv2.resize(seed_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    inner = cv2.erode(seed_mask, np.ones((7, 7), np.uint8))
    outer = cv2.dilate(seed_mask, np.ones((15, 15), np.uint8))
    gc[outer > 0] = cv2.GC_PR_BGD
    gc[seed_mask > 0] = cv2.GC_PR_FGD
    gc[inner > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST) if k < 1 else out


def grabcut_rect_mask(img: Image, margin: float = 0.04, iters: int = 6, size: int = 600) -> Image:
    """Classic rect-initialised GrabCut: everything outside a thin border margin is
    background, the card is whatever the GMMs + edges separate from it."""
    h, w = img.shape[:2]
    k = min(1.0, size / max(h, w))
    sm = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    sh, sw = sm.shape[:2]
    m = int(margin * min(sh, sw))
    mask = np.zeros((sh, sw), np.uint8)
    cv2.grabCut(sm, mask, (m, m, sw - 2 * m, sh - 2 * m), np.zeros((1, 65)), np.zeros((1, 65)),
                iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST) if k < 1 else fg


# --------------------------------------------------------------------------
# corner refinement by robust line fitting
# --------------------------------------------------------------------------
def refine_quad(quad: Quad, contour: Contour, shape: tuple[int, ...]) -> Quad:
    quad = order_corners(quad)
    pts = contour.reshape(-1, 2).astype(np.float32)
    diag = np.hypot(*shape[:2])
    tol = 0.012 * diag
    lines: list[Line] = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = q - p
        L = np.linalg.norm(d)
        u = d / (L + 1e-9)
        n = np.array([-u[1], u[0]])
        rel = pts - p
        t = rel @ u / (L + 1e-9)
        dist = np.abs(rel @ n)
        sel = pts[(t > 0.12) & (t < 0.88) & (dist < tol)]   # skip rounded corners
        if len(sel) < 15:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((vx, vy, x0, y0))
    new = []
    for i in range(4):
        x = intersect(lines[i - 1], lines[i])               # corner i = side i-1 ∩ side i
        if x is None:
            return quad
        new.append(x)
    new = np.array(new, np.float32)
    # sanity: refinement must not wander far from the initial estimate
    if np.max(np.linalg.norm(new - quad, axis=1)) > 0.05 * diag:
        return quad
    return new


# --------------------------------------------------------------------------
# main detector
# --------------------------------------------------------------------------
def pick_by_consensus(scored: list[Scored], shape: tuple[int, ...], margin: float = 0.12) -> Scored:
    """Among candidates scoring within `margin` of the best, return the one that
    overlaps most (IoU) with the others -- independent hypotheses that agree beat a
    single outlier that merely scores a hair higher (e.g. one that follows a
    plastic sleeve instead of the card)."""
    top = [t for t in scored if t[0] >= scored[0][0] - margin]
    if len(top) < 3:
        return scored[0]
    k = 200.0 / max(shape[:2])
    masks = []
    for _, q, _ in top:
        m = np.zeros((int(shape[0] * k) + 1, int(shape[1] * k) + 1), np.uint8)
        cv2.fillConvexPoly(m, np.rint(order_corners(q) * k).astype(np.int32), 1)
        masks.append(m.astype(bool))
    agree = []
    for i, a in enumerate(masks):
        agree.append(sum((a & b).sum() / max((a | b).sum(), 1) for j, b in enumerate(masks) if j != i))
    return top[int(np.argmax(agree))]


def find_card_corners(img_full: Image, work_size: int = 1000) -> Quad | None:
    h0, w0 = img_full.shape[:2]
    s = work_size / max(h0, w0)
    img = cv2.resize(img_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_full.copy()
    s = img.shape[0] / h0

    cands = list(edge_contours(img))
    for cm in colour_mask(img):
        cands += largest_component_contour(cm)
        try:
            cands += largest_component_contour(grabcut_mask(img, cm))
        except cv2.error:
            pass

    try:
        cands += largest_component_contour(grabcut_rect_mask(img))
    except cv2.error:
        pass

    scored: list[Scored] = []
    for c in cands:
        if len(c) < 20:
            continue
        hull = cv2.convexHull(c)
        quad = hull_to_quad(hull)
        sc = quad_score(quad, cv2.contourArea(hull), img.shape)
        if np.isfinite(sc):
            scored.append((sc, quad, c))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best = pick_by_consensus(scored, img.shape)

    quad = refine_quad(best[1], best[2], img.shape)
    sc2 = quad_score(quad, cv2.contourArea(quad), img.shape)
    if not np.isfinite(sc2):
        quad = order_corners(best[1])
    return order_corners(quad) / s                           # back to full-res coords


def ecc_refine(ref: Image, img: Image, levels: int = 3) -> tuple[npt.NDArray[np.float64], float] | None:
    """Register `img` onto `ref` (both already in the canonical card frame).

    Returns (W, cc): a 3x3 homography with img(W x) ~= ref(x), and the final ECC
    correlation; or None on failure.  Coarse-to-fine, with saturated pixels of `img`
    (glare) excluded so they don't pull the estimate."""
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid = cv2.erode(((ig < 235) * 255).astype(np.uint8), np.ones((15, 15), np.uint8))
    if valid.mean() / 255 < 0.3:                              # nearly all glare / white
        return None

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    W, cc = np.eye(3), 0.0
    for lev in range(levels - 1, -1, -1):                     # coarse -> fine
        s = 0.5 ** lev
        S = np.diag([s, s, 1.0])
        if lev:
            r, i, m = (cv2.resize(x, None, fx=s, fy=s, interpolation=interp)
                       for x, interp in ((rg, cv2.INTER_AREA), (ig, cv2.INTER_AREA), (valid, cv2.INTER_NEAREST)))
        else:
            r, i, m = rg, ig, valid
        try:
            cc, Ws = cv2.findTransformECC(r, i, (S @ W @ np.linalg.inv(S)).astype(np.float32),
                                          cv2.MOTION_HOMOGRAPHY, crit, m, 5)
        except cv2.error:
            return None
        W = np.linalg.inv(S) @ Ws.astype(np.float64) @ S      # back to full-res coordinates
    return W, float(cc)


def canonical_homography(corners: Quad, out_w: int = 1012,
                         keep_portrait: bool = False) -> tuple[npt.NDArray[np.float64], tuple[int, int], Quad]:
    """Homography mapping the raw photo to the canonical card frame.
    By default every card comes out landscape (out_w x out_w/1.586).  With
    keep_portrait=True, cards that are portrait in the photo (e.g. vertical licences)
    come out portrait (out_h x out_w) and upright.  Returns (H, (w, h), corners_used)."""
    out_h = int(round(out_w / ID1_ASPECT))
    wd = np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[3] - corners[2])
    ht = np.linalg.norm(corners[0] - corners[3]) + np.linalg.norm(corners[1] - corners[2])
    size = (out_w, out_h)
    if ht > wd:                                              # card is portrait in the photo
        if keep_portrait:
            size = (out_h, out_w)
        else:
            corners = np.roll(corners, -1, axis=0)
    dst = np.array([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst).astype(np.float64)
    return H, size, corners


def warp_card(img: Image, H: npt.NDArray[np.float64], size: tuple[int, int]) -> Image:
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rectify(img: Image, corners: Quad, out_w: int = 1012, keep_portrait: bool = False) -> tuple[Image, Quad]:
    """Corner-based rectification only (no reference alignment)."""
    H, size, corners = canonical_homography(corners, out_w, keep_portrait)
    return warp_card(img, H, size), corners


def draw_debug(img: Image, corners: Quad) -> Image:
    vis = img.copy()
    t = max(2, int(max(img.shape[:2]) / 400))
    cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), t)
    for i, p in enumerate(corners):
        cv2.circle(vis, tuple(int(v) for v in p), t * 3, (0, 0, 255), -1)
        cv2.putText(vis, "TL TR BR BL".split()[i], tuple(int(v) + 10 for v in p),
                    cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (255, 0, 0), t)
    return cv2.resize(vis, None, fx=1000 / max(vis.shape[:2]), fy=1000 / max(vis.shape[:2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, default="./data/raw/")
    ap.add_argument("output_dir", type=Path, default="./data/rectified/")
    ap.add_argument("--width", type=int, default=1012, help="output card width in px (default 1012 ≈ 300 dpi)")
    ap.add_argument("--keep-portrait", action="store_true",
                    help="portrait cards stay portrait/upright instead of being rotated to landscape")
    ap.add_argument("--debug", action="store_true", help="save corner-overlay images to OUTPUT_DIR/debug/")
    ap.add_argument("--no-align", action="store_true",
                    help="skip ECC alignment to the card's S_D reference (corner-based rectification only)")
    ap.add_argument("--max-shift", type=float, default=100.0,
                    help="reject the ECC correction if it moves a card corner by more than this many px "
                         "in the rectified image (default 100)")
    a = ap.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir: Path = a.output_dir / "debug"
    if a.debug:
        debug_dir.mkdir(exist_ok=True)
    files = sorted(p for p in a.input_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No images found in {a.input_dir}")

    # The S_D image of each card is its reference; process those first so the other
    # images of the card can be registered to them.
    refs: dict[str, Path] = {}
    for p in files:
        m = FILE_RE.match(p.stem)
        if m and m["blur"] == "S" and m["glare"] == "D":
            refs.setdefault(m["card"], p)
    files.sort(key=lambda p: p not in refs.values())          # stable: refs first

    failed: list[str] = []
    ref_rect: dict[str, Image] = {}                           # card -> rectified reference
    log: list[dict] = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)          # NOTE: ignores EXIF rotation for PNG (n/a)
        if img is None:
            print(f"[skip] cannot read {p.name}")
            failed.append(p.name)
            continue
        corners = find_card_corners(img)
        if corners is None:
            print(f"[FAIL] no card found in {p.name}")
            failed.append(p.name)
            continue

        H, size, corners = canonical_homography(corners, a.width, a.keep_portrait)
        out = warp_card(img, H, size)
        m = FILE_RE.match(p.stem)
        card = m["card"] if m else None
        status, shift, cc = "corners-only", float("nan"), float("nan")

        if card in refs and refs[card] == p:
            ref_rect[card] = out
            status = "reference"
        elif a.no_align or card not in refs:
            pass
        elif card not in ref_rect:
            status = "NO-REFERENCE (reference image failed)"
        elif ref_rect[card].shape != out.shape:
            status = "NOT-ALIGNED (size/orientation differs from reference)"
        else:
            res = ecc_refine(ref_rect[card], out)
            if res is None:
                status = "NOT-ALIGNED (ECC failed)"
            else:
                W, cc = res
                w_, h_ = size
                pts = np.array([[0, 0], [w_ - 1, 0], [w_ - 1, h_ - 1], [0, h_ - 1]], np.float64)
                moved = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), W).reshape(-1, 2)
                shift = float(np.linalg.norm(moved - pts, axis=1).max())
                if cc < 0.4 or shift > a.max_shift:
                    status = "NOT-ALIGNED (ECC rejected)"
                else:
                    out = warp_card(img, np.linalg.inv(W) @ H, size)   # single warp of the raw photo
                    status = "aligned"

        cv2.imwrite(str(a.output_dir / f"{p.stem}_rectified.png"), out)
        if a.debug:
            cv2.imwrite(str(debug_dir / f"{p.stem}_debug.jpg"), draw_debug(img, corners))
        extra = f"  corner-shift={shift:.2f}px cc={cc:.3f}" if np.isfinite(shift) else ""
        print(f"[ok]   {p.name:45s} {status}{extra}")
        log.append(dict(name=p.name, status=status, max_corner_shift_px=shift, ecc_cc=cc))

    with open(a.output_dir / "alignment_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "max_corner_shift_px", "ecc_cc"])
        w.writeheader()
        w.writerows(log)

    print(f"\nDone: {len(files) - len(failed)}/{len(files)} succeeded.")
    n_aligned = sum(r["status"] == "aligned" for r in log)
    n_bad = [r["name"] for r in log if r["status"].startswith(("NOT-ALIGNED", "NO-REFERENCE"))]
    print(f"Aligned to reference: {n_aligned}; refs: {sum(r['status'] == 'reference' for r in log)}; "
          f"needs a look: {len(n_bad)}")
    if n_bad:
        print("Not aligned:", ", ".join(n_bad))
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
