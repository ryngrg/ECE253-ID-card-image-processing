#!/usr/bin/env python3
"""
Evaluate glare removal: SSIM and OCR agreement with the glare-free reference.

For every usable glare image in out/pairs.csv (made by prepare_data.py) it compares
the image (untreated, or your processed version) with the card's S_D reference:

  ssim        SSIM on grayscale, whole image
  ssim_glare  SSIM averaged only inside the glare region.  The region is where the
              UNTREATED image is much brighter than the reference, so it is the same
              for every method and independent of any glare detector
  ssim_outside  SSIM away from the glare.  Should stay high; if it is already low for the
              untreated images, the SSIM floor comes from misalignment / exposure
              differences, not from glare.  Also a "do no harm" check for methods
  cer         character error rate of the OCR text vs the OCR text of the reference
              (clipped to 1.0; lower is better)
  word_f1     bag-of-words F1 between the two OCR outputs (higher is better); unlike
              CER it ignores reading order, which is unstable on ID layouts

If you typed the true text of each card (ground_truth/{card_id}.txt) it also reports
  f1_gt, recall_gt   word F1 / recall of the OCR output against the TRUE text
which does not depend on how well the engine reads the clean reference.

Usage
  python src/common/evaluate.py --refs-only                        # OCR the 22 references: look at the
                                                        # text, create ground-truth templates
  python src/common/evaluate.py                                    # baseline: untreated glare images
  python src/common/evaluate.py --treated-dir out/classical --name classical
  python src/common/evaluate.py --treated-dir out/dl_ft --name dl_ft --splits val,test
  python src/common/evaluate.py --combine                          # one table from all runs
  add --engine paddleocr (default, the engine we report) | tesseract, --lang, --save-ocr
  add --no-ocr for a fast SSIM-only run (no OCR columns)
  OCR results are cached in results/ocr_cache/, so repeated runs only OCR new/changed images

Processed images must have the SAME FILENAMES as the rectified inputs
(e.g. B01_S_G_0_rectified.png) and the same size.

Requirements: requirements.txt (PaddleOCR 3.x; for --engine tesseract also the tesseract program).
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

LINE_BUCKET = 30  # px (in the 1012-px-wide card frame) used to order words top-to-bottom


# --------------------------------------------------------------------------
# OCR engines.  ocr(img_bgr) -> list of items (left, center_y, word, (x0, y0, x1, y1))
# --------------------------------------------------------------------------
def _bbox(poly):
    xs, ys = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _words_from_lines(lines):
    """lines: [(text, bbox)] -> word items; all words of a line share its position."""
    return [(b[0], (b[1] + b[3]) / 2, w, b) for text, b in lines for w in text.split()]


def make_ocr(engine: str, lang: str | None):
    if engine == "tesseract":
        import pytesseract
        lang = lang or "eng"

        def ocr(img):
            d = pytesseract.image_to_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lang=lang,
                                          config="--psm 11", output_type=pytesseract.Output.DICT)
            items = []
            for i, t in enumerate(d["text"]):
                if t.strip() and float(d["conf"][i]) > 0:
                    x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                    items.append((x, y + h / 2, t, (x, y, x + w, y + h)))
            return items
        return ocr

    if engine == "paddleocr":
        from paddleocr import PaddleOCR
        lang = lang or "en"
        model = PaddleOCR(lang=lang, use_doc_orientation_classify=False,  # PaddleOCR 3.x
                          use_doc_unwarping=False, use_textline_orientation=False)

        def ocr(img):
            lines = []
            for res in model.predict(img):
                d = res if isinstance(res, dict) else res.json["res"]
                polys = d.get("rec_polys", d.get("dt_polys"))
                lines += [(t, _bbox(p)) for t, p in zip(d["rec_texts"], polys)]
            return _words_from_lines(lines)
        return ocr

    raise SystemExit(f"unknown engine {engine}")


def normalize(tokens) -> list[str]:
    """Lowercase, keep letters/digits only (any script), drop empty tokens."""
    words = (re.sub(r"[\W_]+", "", t.lower()) for t in tokens)
    return [w for w in words if w]


def _ordered(items):
    return sorted(items, key=lambda t: (int(t[1] // LINE_BUCKET), t[0]))


def to_words(items) -> list[str]:
    return normalize(t[2] for t in _ordered(items))


def to_lines(items) -> list[str]:
    lines: dict[int, list[str]] = {}
    for t in _ordered(items):
        lines.setdefault(int(t[1] // LINE_BUCKET), []).append(t[2])
    return [" ".join(v) for _, v in sorted(lines.items())]


def dump_ocr(out_dir: Path, stem: str, img, items):
    """Save the OCR text and an image with the detected boxes drawn on it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text("\n".join(to_lines(items)) + "\n", encoding="utf-8")
    vis = img.copy()
    for t in items:
        x0, y0, x1, y1 = (int(v) for v in t[3])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 2)
    cv2.imwrite(str(out_dir / f"{stem}_boxes.png"), vis)


def load_gt(gt_dir: Path, card_id: str):
    p = gt_dir / f"{card_id}.txt"
    return normalize(p.read_text(encoding="utf-8").split()) if p.exists() else None


def ocr_file(ocr, path, cache_dir: Path | None):
    """OCR the image file at `path`; reuse a stored result if the file is unchanged.
    Cache key = path + size + modification time (delete results/ocr_cache if you change
    OCR settings other than --engine / --lang, which have their own cache folders)."""
    path = Path(path)
    f = None
    if cache_dir is not None:
        st = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        f = cache_dir / f"{key}.json"
        if f.exists():
            return [(l, c, t, tuple(b)) for l, c, t, b in json.loads(f.read_text())]
    items = ocr(cv2.imread(str(path)))
    if f is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([(float(l), float(c), t, [float(v) for v in b]) for l, c, t, b in items]))
    return items


# --------------------------------------------------------------------------
# Text metrics
# --------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref_words: list[str], hyp_words: list[str]) -> float:
    ref, hyp = " ".join(ref_words), " ".join(hyp_words)
    return min(1.0, edit_distance(ref, hyp) / len(ref))


def _overlap(ref_words, hyp_words) -> int:
    return sum((Counter(ref_words) & Counter(hyp_words)).values())


def word_f1(ref_words: list[str], hyp_words: list[str]) -> float:
    if not ref_words or not hyp_words:
        return 0.0
    o = _overlap(ref_words, hyp_words)
    if o == 0:
        return 0.0
    p, r = o / len(hyp_words), o / len(ref_words)
    return 2 * p * r / (p + r)


def word_recall(ref_words: list[str], hyp_words: list[str]) -> float:
    return _overlap(ref_words, hyp_words) / len(ref_words) if ref_words else float("nan")


# --------------------------------------------------------------------------
# Image metrics
# --------------------------------------------------------------------------
def glare_region(untreated_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    """Pixels where the untreated image is much brighter than the reference."""
    a = cv2.GaussianBlur(untreated_gray, (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(ref_gray, (0, 0), 3).astype(np.float32)
    m = ((a - b) > 40).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    return m.astype(bool)


def ssim_scores(ref_gray, img_gray, mask):
    """(whole image, inside glare region, far outside glare region)"""
    s, smap = structural_similarity(ref_gray, img_gray, data_range=255, full=True)
    s_in = float(smap[mask].mean()) if mask.sum() >= 100 else float("nan")
    outside = ~cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    s_out = float(smap[outside].mean()) if outside.sum() >= 100 else float("nan")
    return float(s), s_in, s_out


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def combine(out_dir: Path):
    files = sorted(out_dir.glob("*_summary.csv"))
    if not files:
        raise SystemExit(f"No *_summary.csv files in {out_dir}")
    table = pd.concat([pd.read_csv(f) for f in files])
    print(table.round(3).to_string(index=False))
    table.to_csv(out_dir / "all_methods.csv", index=False)
    print(f"\nSaved {out_dir / 'all_methods.csv'}")


def run_refs_only(a, pairs, ocr):
    """OCR every reference image: save text + boxes, create ground-truth templates, and
    score the reference OCR against ground truth where it exists."""
    out = a.out_dir / "ocr_text" / a.engine
    a.gt_dir.mkdir(parents=True, exist_ok=True)
    rows, created = [], 0
    for r in pairs.drop_duplicates("card_id").itertuples():
        img = cv2.imread(r.ref_path)
        items = ocr_file(ocr, r.ref_path, a.cache_dir)
        words = to_words(items)
        dump_ocr(out, Path(r.ref_path).stem, img, items)
        gt_path = a.gt_dir / f"{r.card_id}.txt"
        f1 = rec = float("nan")
        if gt_path.exists():
            gt = load_gt(a.gt_dir, r.card_id)
            f1, rec = word_f1(gt, words), word_recall(gt, words)
        else:  # template pre-filled with this engine's output, to be corrected by hand
            gt_path.write_text("\n".join(to_lines(items)) + "\n", encoding="utf-8")
            created += 1
        rows.append(dict(card_id=r.card_id, n_words=len(words), f1_vs_gt=f1, recall_vs_gt=rec))
    res = pd.DataFrame(rows)
    res.to_csv(a.out_dir / f"reference_ocr_{a.engine}.csv", index=False)
    print(res.round(3).to_string(index=False))
    if res.f1_vs_gt.notna().any():
        print(f"\nReference OCR vs ground truth ({a.engine}): mean word F1 {res.f1_vs_gt.mean():.3f}, "
              f"mean recall {res.recall_vs_gt.mean():.3f}  (n={res.f1_vs_gt.notna().sum()} cards)")
    print(f"\nOCR text + boxes: {out}/")
    if created:
        print(f"Created {created} ground-truth templates in {a.gt_dir}/ (pre-filled with this "
              f"engine's output: correct them against the card image).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--treated-dir", type=Path, default=None,
                    help="folder with processed images; omit to evaluate the untreated images")
    ap.add_argument("--name", default=None, help="label for this run (default: untreated / folder name)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--engine", default="paddleocr", choices=["paddleocr", "tesseract"])
    ap.add_argument("--lang", default=None)
    ap.add_argument("--gt-dir", default="ground_truth", type=Path)
    ap.add_argument("--out-dir", default="results/glare/whole_card", type=Path)
    ap.add_argument("--cache-root", default="results/ocr_cache", type=Path)
    ap.add_argument("--refs-only", action="store_true",
                    help="only OCR the reference images (text, boxes, ground-truth templates)")
    ap.add_argument("--save-ocr", action="store_true", help="save OCR text + boxes of every evaluated image")
    ap.add_argument("--no-cache", action="store_true", help="do not reuse stored OCR results")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR (SSIM columns only; fast)")
    ap.add_argument("--combine", action="store_true", help="print one table from all saved runs and exit")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.cache_dir = None if a.no_cache else a.cache_root / f"{a.engine}_{a.lang or 'default'}"
    if a.combine:
        return combine(a.out_dir)

    pairs = pd.read_csv(a.pairs)
    ocr = None if a.no_ocr and not a.refs_only else make_ocr(a.engine, a.lang)
    if a.refs_only:
        return run_refs_only(a, pairs, ocr)

    name = a.name or (a.treated_dir.name if a.treated_dir else "untreated")
    pairs = pairs[pairs.usable & pairs.split.isin(a.splits.split(","))].reset_index(drop=True)
    print(f"Evaluating '{name}' ({a.engine}) on {len(pairs)} images (splits: {a.splits})")

    ref_cache: dict[str, tuple] = {}
    rows = []
    for k, r in enumerate(pairs.itertuples(), 1):
        if r.ref_path not in ref_cache:
            ref = cv2.imread(r.ref_path)
            ref_cache[r.ref_path] = (cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
                                     to_words(ocr_file(ocr, r.ref_path, a.cache_dir)) if ocr else [], ref.shape)
        ref_gray, ref_words, ref_shape = ref_cache[r.ref_path]
        if ocr and len(ref_words) < 5:
            print(f"[warn] reference OCR found only {len(ref_words)} words: {Path(r.ref_path).name}")
            continue

        untreated = cv2.imread(r.path)
        img_path = (a.treated_dir / Path(r.path).name) if a.treated_dir else Path(r.path)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] missing/unreadable: {img_path}")
            continue
        if img.shape != ref_shape:
            print(f"[warn] size {img.shape[:2]} != reference {ref_shape[:2]}: {img_path.name}")
            continue

        mask = glare_region(cv2.cvtColor(untreated, cv2.COLOR_BGR2GRAY), ref_gray)
        s, s_glare, s_out = ssim_scores(ref_gray, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
        row = dict(image=Path(r.path).name, card_id=r.card_id, condition=r.condition, split=r.split,
                   ssim=s, ssim_glare=s_glare, ssim_outside=s_out, glare_frac=mask.mean())
        if ocr:
            items = ocr_file(ocr, img_path, a.cache_dir)
            hyp_words = to_words(items)
            if a.save_ocr:
                dump_ocr(a.out_dir / "ocr_text" / name, Path(r.path).stem, img, items)
            gt = load_gt(a.gt_dir, r.card_id)
            row.update(
                cer=cer(ref_words, hyp_words) if hyp_words else 1.0,
                word_f1=word_f1(ref_words, hyp_words),
                f1_gt=word_f1(gt, hyp_words) if gt else np.nan,
                recall_gt=word_recall(gt, hyp_words) if gt else np.nan,
                n_ref_words=len(ref_words), n_hyp_words=len(hyp_words))
        rows.append(row)
        if k % 10 == 0:
            print(f"  {k}/{len(pairs)}")

    res = pd.DataFrame(rows)
    res.to_csv(a.out_dir / f"{name}_per_image.csv", index=False)

    agg = dict(n=("ssim", "size"), ssim=("ssim", "mean"), ssim_glare=("ssim_glare", "mean"),
               ssim_outside=("ssim_outside", "mean"), glare_frac=("glare_frac", "mean"))
    if ocr:
        agg.update(cer=("cer", "mean"), word_f1=("word_f1", "mean"))
    if ocr and res.f1_gt.notna().any():
        agg.update(f1_gt=("f1_gt", "mean"), recall_gt=("recall_gt", "mean"))
    summary = res.groupby(["split", "condition"]).agg(**agg).reset_index()
    summary.insert(0, "method", name)
    summary.to_csv(a.out_dir / f"{name}_summary.csv", index=False)

    print(f"\n=== {name} ===")
    print(summary.round(3).to_string(index=False))
    if ocr:
        print(f"\nreference OCR words per card: median {np.median([len(v[1]) for v in ref_cache.values()]):.0f}")
    print(f"Saved {a.out_dir}/{name}_per_image.csv and {name}_summary.csv")


if __name__ == "__main__":
    main()
