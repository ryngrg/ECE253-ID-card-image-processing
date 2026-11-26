#!/usr/bin/env python3
"""
External test set: the MIDV-2020 photos labelled "highlight present" (README "Data").

MIDV-2020 (Bulatov et al., 2021; CC BY-SA 2.5) has 1000 mock identity documents, 10 types x 100.
Each document has a digital template (with field rectangles + field text), a flatbed scan and a
phone photo.  Photos 60-69 of every type were taken with a highlight present -> 100 glare photos
of 100 different documents, none of which was used for tuning or training.

Download (public mirror, no form needed):
  mkdir -p data/external/midv2020 && cd data/external/midv2020
  for f in photo.tar templates.tar scan_upright.tar; do curl -O ftp://smartengines.com/midv-2020/dataset/$f; done
  (checksums: ftp://smartengines.com/midv-2020/md5.txt; only the highlight images + annotations are needed:)
  for t in alb_id aze_passport esp_id est_id fin_id grc_passport lva_passport rus_internalpassport srb_passport svk_id;
    do for n in 60 61 62 63 64 65 66 67 68 69; do echo images/$t/$n.jpg; done; done > highlight_list.txt
  mkdir -p photo scan_upright templates && tar -xf templates.tar -C templates
  for s in photo scan_upright; do tar -xf $s.tar -C $s -T highlight_list.txt; tar -xf $s.tar -C $s annotations; done

What this script does, per document
  1. photo and scan are warped from their annotated doc_quad to the canonical frame at our resolution
     (~11.8 px/mm: ID-1 cards 1012 px wide, like data/rectified; passports 125 mm -> 1478 px wide;
     height from the template's aspect ratio);
  2. the photo is ECC-aligned to the scan (rectify_ids.ecc_refine, glare masked out; kept only if it
     raises the correlation and moves no corner > 15 px, else the quad warp alone) -> the scan is the
     glare-free reference, as S_D is for our cards;
  3. the template's field rectangles are mapped to that frame (template -> scan ECC, quad-only if
     ECC is rejected).  Kept as fixed OCR boxes: orientation 0, text that is only Latin letters/digits
     after folding accents (drops Greek / Cyrillic / Azerbaijani schwa fields), >= 3 characters,
     not overlapping another kept box;
  4. glare mask: the scan's colours differ from the photo's (other device, no shading), so it is first
     tone-matched to the photo (robust per-channel gain/offset + a smooth additive illumination field,
     sigma 150 px, fitted outside outliers), then glare = photo brighter than the matched scan by > 40 grey levels
     (same rule as evaluate.glare_region).

Outputs
  data/midv/rectified/{doc}_{nn}.png        photo, aligned to the scan frame   ("untreated")
  data/midv/reference/{doc}_{nn}.png        rectified scan                      (ceiling run)
  data/midv/glare_masks/{doc}_{nn}.png      glare region (255) used for the core/halo/outside zones
  data/midv/overlays/{doc}_{nn}.jpg         photo | matched scan | glare + boxes  (visual check)
  out/pairs_midv.csv                        same columns as out/pairs.csv (split = midv, condition = S_G)
  results/layouts/box_layout_midv.csv               fixed boxes + targets (box_ocr.py --layout)
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common.rectify_ids import ecc_refine

TYPES = ["alb_id", "aze_passport", "esp_id", "est_id", "fin_id", "grc_passport", "lva_passport",
         "rus_internalpassport", "srb_passport", "svk_id"]
NUMS = range(60, 70)          # "highlight present" photos (Samsung S10: 65-69, iPhone XR: 60-64)
PX_PER_MM = 1012 / 85.6       # our rectified resolution
MIN_LEN = 3
GLARE_DIFF = 40               # as evaluate.glare_region


def fold(s: str) -> str:
    """Lowercase, strip accents, keep letters/digits only ('Rožňava' -> 'roznava')."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", s.lower())


def regions(root: Path, src: str, doc: str, n: int):
    meta = json.loads((root / src / "annotations" / f"{doc}.json").read_text())["_via_img_metadata"]
    return next(v for v in meta.values() if v["filename"] == f"{n:02d}.jpg")["regions"]


def doc_quad(regs) -> np.ndarray:
    a = next(r for r in regs if r["region_attributes"].get("field_name") == "doc_quad")["shape_attributes"]
    return np.float32(list(zip(a["all_points_x"], a["all_points_y"])))


def warp(img, quad, size):
    w, h = size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def safe_ecc(ref, img, max_shift=15.0):
    """ecc_refine, accepted only if it raises the ECC correlation over the identity (quad-only)
    warp and moves no corner by more than max_shift px (ECC sometimes diverges when glare or
    holograms dominate).  Returns (W or identity, cc_before, cc_after, accepted)."""
    h, w = ref.shape[:2]
    rg, ig = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (ref, img))
    mask = cv2.erode(((ig < 235) * 255).astype(np.uint8), np.ones((15, 15), np.uint8))
    cc0 = float(cv2.computeECC(rg, ig, mask))
    res = ecc_refine(ref, img)
    if res is None:
        return np.eye(3), cc0, np.nan, False
    corners = np.float32([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])
    shift = np.abs(cv2.perspectiveTransform(corners, res[0]) - corners).max()
    ok = res[1] > cc0 + 0.005 and shift <= max_shift
    return (res[0] if ok else np.eye(3)), cc0, res[1], bool(ok)


def norm_conv(v, w, sigma, f=8):
    """Normalised convolution (weighted Gaussian smoothing), computed at 1/f resolution."""
    vs, ws = (cv2.resize(x, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA) for x in (v * w, w))
    out = cv2.GaussianBlur(vs, (0, 0), sigma / f) / np.maximum(cv2.GaussianBlur(ws, (0, 0), sigma / f), 1e-3)
    return cv2.resize(out, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)


def match_tone(photo, scan, iters=4, sigma=150, inlier=25):
    """Scan re-coloured to look like the photo without its glare: per channel photo ~ a*scan + b
    (least squares on inliers) plus a smooth additive field (normalised convolution of the
    residual over inliers).  Glare pixels become outliers and are excluded from the fit.
    sigma = 150 px: with 40 px the field absorbed broad glare stripes; with no field at all,
    shading differences between photo and scan leak into the mask."""
    P = cv2.GaussianBlur(photo.astype(np.float32), (0, 0), 2)
    S = cv2.GaussianBlur(scan.astype(np.float32), (0, 0), 2)
    valid = (photo.min(axis=2) < 245).astype(np.float32)
    w = valid.copy()
    pred = S.copy()
    for _ in range(iters):
        for c in range(3):
            m = w > 0
            A = np.stack([S[..., c][m], np.ones(m.sum(), np.float32)], 1)
            ab, *_ = np.linalg.lstsq(A, P[..., c][m], rcond=None)
            lin = S[..., c] * ab[0] + ab[1]
            pred[..., c] = lin + norm_conv(P[..., c] - lin, w, sigma)
        resid = (P - pred).mean(axis=2)
        w = valid * (np.abs(resid) < inlier)
    return np.clip(pred, 0, 255).astype(np.uint8)


def glare_mask(photo, matched):
    a = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    m = ((a - b) > GLARE_DIFF).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool)


def overlap(b1, b2):
    """Intersection over the smaller box area."""
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    a = min((b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return ix * iy / max(a, 1e-6)


def canonical_size(doc: str, tmpl) -> tuple[int, int]:
    """Canonical frame at our resolution: width from the physical size (ID-1 card 85.6 mm, passport
    page 125 mm), height from the template's aspect ratio."""
    th, tw = tmpl.shape[:2]
    W = round((125.0 if "passport" in doc else 85.6) * PX_PER_MM)
    return W, round(W * th / tw)


def template_boxes(root: Path, doc: str, n: int, tmpl, scan):
    """Template field rectangles mapped to the scan frame (template -> scan ECC, quad-only if the
    guard rejects it), with the keep/drop rule.  Returns (boxes, template_ecc_used)."""
    H, W = scan.shape[:2]
    th, tw = tmpl.shape[:2]
    tm = cv2.resize(tmpl, (W, H), interpolation=cv2.INTER_AREA)
    Wt, _, _, ok_t = safe_ecc(scan, tm)
    Wi = np.linalg.inv(Wt)
    s = W / tw
    boxes = []
    for r in regions(root, "templates", doc, n):
        sa, ra = r["shape_attributes"], r["region_attributes"]
        val = ra.get("value", "")
        if sa["name"] != "rect" or not val:
            continue
        corners = np.float32([[[sa["x"] * s, sa["y"] * s], [(sa["x"] + sa["width"]) * s, sa["y"] * s],
                               [(sa["x"] + sa["width"]) * s, (sa["y"] + sa["height"]) * s],
                               [sa["x"] * s, (sa["y"] + sa["height"]) * s]]])
        p = cv2.perspectiveTransform(corners, Wi)[0]
        b = [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
        target = fold(val)
        latin = bool(re.fullmatch(r"[a-z0-9]+", target)) and not re.search(r"[^\x00-\x7f]", target)
        reason = ("orientation" if str(ra.get("orientation", "0")) != "0" else
                  "script" if not latin else "short" if len(target) < MIN_LEN else "")
        boxes.append(dict(field=ra.get("field_name", ""), value=val, target=target, b=b, reason=reason))
    kept = [x for x in boxes if not x["reason"]]
    for x in kept:
        if any(overlap(x["b"], y["b"]) > 0.3 for y in kept if y is not x):
            x["reason"] = "overlap"
    return boxes, ok_t


def layout_rows(cid: str, doc: str, boxes) -> list[dict]:
    """Rows of the box layout CSV (box_ocr.py --layout)."""
    return [dict(card_id=cid, doc_type=doc, box=k, field=x["field"], value=x["value"],
                 x0=x["b"][0], y0=x["b"][1], x1=x["b"][2], y1=x["b"][3], target=x["target"],
                 labelled=not x["reason"], drop_reason=x["reason"],
                 digits=sum(c.isdigit() for c in x["target"]) >= 4) for k, x in enumerate(boxes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/midv2020", type=Path)
    ap.add_argument("--out", default="data/midv", type=Path)
    ap.add_argument("--pairs-out", default="out/pairs_midv.csv", type=Path)
    ap.add_argument("--layout-out", default="results/layouts/box_layout_midv.csv", type=Path)
    a = ap.parse_args()
    for d in ("rectified", "reference", "glare_masks", "overlays"):
        (a.out / d).mkdir(parents=True, exist_ok=True)

    pairs, layout = [], []
    for doc in TYPES:
        for n in NUMS:
            cid = f"{doc}_{n:02d}"
            tmpl = cv2.imread(str(a.root / "templates" / "images" / doc / f"{n:02d}.jpg"))
            photo_raw = cv2.imread(str(a.root / "photo" / "images" / doc / f"{n:02d}.jpg"))
            scan_raw = cv2.imread(str(a.root / "scan_upright" / "images" / doc / f"{n:02d}.jpg"))
            W, H = canonical_size(doc, tmpl)
            scan = warp(scan_raw, doc_quad(regions(a.root, "scan_upright", doc, n)), (W, H))
            photo = warp(photo_raw, doc_quad(regions(a.root, "photo", doc, n)), (W, H))

            # photo -> scan frame
            Wp, cc0, cc1, ok_p = safe_ecc(scan, photo)
            if ok_p:
                photo = cv2.warpPerspective(photo, Wp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                            borderMode=cv2.BORDER_REPLICATE)
            boxes, ok_t = template_boxes(a.root, doc, n, tmpl, scan)
            kept = [x for x in boxes if not x["reason"]]
            layout += layout_rows(cid, doc, boxes)

            matched = match_tone(photo, scan)
            gm = glare_mask(photo, matched)
            p_path, r_path, m_path = (a.out / d / f"{cid}.png" for d in ("rectified", "reference", "glare_masks"))
            cv2.imwrite(str(p_path), photo)
            cv2.imwrite(str(r_path), scan)
            cv2.imwrite(str(m_path), gm.astype(np.uint8) * 255)

            vis = photo.copy()
            vis[gm] = (0.5 * vis[gm] + [0, 0, 127]).astype(np.uint8)
            for x in boxes:
                col = (0, 200, 0) if not x["reason"] else (160, 160, 160)
                cv2.rectangle(vis, (int(x["b"][0]), int(x["b"][1])), (int(x["b"][2]), int(x["b"][3])), col, 1)
            cv2.imwrite(str(a.out / "overlays" / f"{cid}.jpg"), np.hstack([photo, matched, vis]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

            clipped = int((photo.min(axis=2) >= 248).sum())
            pairs.append(dict(card_id=cid, doc_type=doc, num=n, device="iphone_xr" if n < 65 else "samsung_s10",
                              path=str(p_path), ref_path=str(r_path), glare_mask_path=str(m_path),
                              condition="S_G", split="midv", ecc_cc_quad=cc0, ecc_cc=cc1 if ok_p else cc0,
                              ecc_used=ok_p, template_ecc_used=ok_t,
                              glare_frac=float(gm.mean()), clipped_px=clipped, usable=True, problem=""))
            print(f"{cid:26s} {W}x{H}  ecc photo {cc0:.3f}->{cc1:.3f} {'used' if ok_p else 'rejected'}  "
                  f"template ecc {'used' if ok_t else 'rejected'}  "
                  f"glare {gm.mean():.3f}  boxes {len(kept)}/{len(boxes)}", flush=True)

    P = pd.DataFrame(pairs)
    L = pd.DataFrame(layout)
    a.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(a.pairs_out, index=False)
    L.to_csv(a.layout_out, index=False)
    print(f"\n{len(P)} photos ({P.usable.sum()} usable), median glare fraction {P.glare_frac.median():.3f}; "
          f"{L.labelled.sum()} labelled boxes of {len(L)} "
          f"(dropped: {L[~L.labelled].drop_reason.value_counts().to_dict()})")
    print(f"Saved {a.pairs_out}, {a.layout_out}, {a.out}/")


if __name__ == "__main__":
    main()
