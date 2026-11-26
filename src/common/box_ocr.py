#!/usr/bin/env python3
"""
Fixed-box, recognition-only OCR evaluation per glare zone (the main OCR metric).

Why.  Whole-card OCR (evaluate.py) mixes two sources of noise into every comparison: the text
DETECTOR splits / merges boxes differently on near-identical images, and the glare covers ~10% of
the card, so its effect is diluted by the other 90%.  Here
  * the text boxes are fixed per card: the line boxes PaddleOCR detects on the glare-free S_D
    reference.  All images are aligned to that reference, so the same boxes apply to the
    untreated image and to every treated version of it;
  * each box is cropped and only RECOGNISED (no detection), so if a method leaves a box's pixels
    unchanged, its reading is bit-identical;
  * each box is labelled with its true text: the substring of the card's ground truth
    (ground_truth/{card}.txt, one line at a time) that best matches the reference reading
    (approximate substring matching; boxes with no close match, or < 3 characters, are dropped);
  * in every glare image each box is put in one zone, from the UNTREATED image (same for all
    methods):  core = overlaps the clipped core,  halo = overlaps the rest of the glare region,
    outside = elsewhere  (>= 10% of the box area).

Scores (characters = lowercase letters/digits only, spaces and punctuation ignored)
  char_acc_{zone}   1 - (sum of edit distances / sum of target lengths) over the boxes of the zone
                    in one image (higher is better; can be < 0 for long hallucinations, clipped at 0)
  exact_{zone}      fraction of boxes read exactly right
  zones: core, halo, outside, glare (= core + halo), all;  digits_glare / digits_all use only boxes
  whose target has >= 4 digits (dates, ID numbers)

Runs (results/{glare,blur,joint}/box_{engine}[_tag]/{run}_per_image.csv, {run}_boxes.csv, summary.csv)
  The layout and the {run}_boxes.csv readings contain card text: they are written with common.privacy.save_csv,
  which blanks the text of our real cards in the tracked file and keeps the full table in private/ (local).
  untreated, reference (ceiling: the clean S_D crop), ref_jitter (noise floor proxy: the reference
  with each box shifted by up to 2 px, Gaussian noise sigma 2 and JPEG q 90; the change between
  reference and ref_jitter is the smallest effect the recogniser can resolve; a second real clean
  photo would be the proper floor), and every folder of out/ given in --runs.

External set (MIDV-2020, see prepare_midv.py): fixed boxes = annotated fields, zones from the
precomputed glare masks (column glare_mask_path of the pairs file), accents folded:
  python src/common/box_ocr.py --pairs out/pairs_midv.csv --layout results/layouts/box_layout_midv.csv --splits midv \
      --tag midv --out-root out/midv --runs classical,dl_zeroshot,dl_ft --fold-accents --pad 2
  -> results/glare/box_{engine}_midv/

Usage
  python src/common/box_ocr.py                                    # PaddleOCR recogniser, default runs
  python src/common/box_ocr.py --engine tesseract                 # Tesseract --psm 7 on each line crop
  python src/common/box_ocr.py --only --runs classical --extra out/chain/restormer:restormer
  then  python src/common/compare_runs.py --results-dir results/glare/box_paddle --base untreated --new dl_ft --splits test

Engines
  paddle     PaddleOCR TextRecognition, PP-OCRv6_medium_rec (the recogniser of the full pipeline)
  tesseract  Tesseract 5, --psm 7 (single text line), crop upscaled 2x (cubic) + 10 px replicated border
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from glare.classical_glare import DEFAULTS, detect_core
from common.evaluate import edit_distance, glare_region, ocr_file
from common import privacy

PAD = 0            # px of context around each box (lines are tightly packed: a pad pulls in
                   # strokes of the neighbouring lines and wrecks the reading of long lines)
OVERLAP = 0.10     # zone membership threshold (fraction of the box area)
MAX_LABEL_DIST = 0.34  # max normalised edit distance between reference reading and GT substring
MIN_LEN = 3        # boxes with shorter targets are dropped (field numbers such as "2", "4a")
ZONES = ["core", "halo", "outside", "glare", "all", "digits_glare", "digits_all"]
DEFAULT_RUNS = "classical,classical_veil,classical_inpaint,classical_telea,classical_decay,dl_zeroshot,dl_ft"


FOLD = False      # --fold-accents: strip diacritics before scoring ('Rožňava' == 'roznava')


def norm(s: str) -> str:
    if FOLD:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", s.lower())


# --------------------------------------------------------------------------
# Box layout per card
# --------------------------------------------------------------------------
def best_substring(pattern: str, text: str):
    """Approximate substring match (Sellers): the substring of `text` with the smallest edit
    distance to `pattern`.  Returns (distance, substring)."""
    m, n = len(pattern), len(text)
    prev = [0] * (n + 1)
    prev_start = list(range(n + 1))
    for i in range(1, m + 1):
        cur, cur_start = [i] + [0] * n, [0] + [0] * n
        for j in range(1, n + 1):
            opts = ((prev[j - 1] + (pattern[i - 1] != text[j - 1]), prev_start[j - 1]),
                    (prev[j] + 1, prev_start[j]),
                    (cur[j - 1] + 1, cur_start[j - 1]))
            cur[j], cur_start[j] = min(opts)
        prev, prev_start = cur, cur_start
    j = int(np.argmin(prev))
    return prev[j], text[prev_start[j]:j]


def build_layout(pairs: pd.DataFrame, gt_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """One row per (card, box): reference line box, reference reading, GT target."""
    ocr = None
    rows = []
    for card, ref_path in pairs.drop_duplicates("card_id")[["card_id", "ref_path"]].itertuples(index=False):
        gt_file = gt_dir / f"{card}.txt"
        if not gt_file.exists():
            print(f"[warn] no ground truth for {card}: skipped")
            continue
        gt_lines = [norm(l) for l in gt_file.read_text(encoding="utf-8").splitlines() if norm(l)]
        try:
            items = ocr_file(None, ref_path, cache_dir)
        except TypeError:  # not cached yet: run the full PaddleOCR pipeline once
            if ocr is None:
                from common.evaluate import make_ocr
                ocr = make_ocr("paddleocr", None)
            items = ocr_file(ocr, ref_path, cache_dir)
        lines: dict[tuple, list[str]] = {}
        for _l, _c, t, b in items:  # words of one detected line share its box
            lines.setdefault(tuple(b), []).append(t)
        for k, (b, words) in enumerate(sorted(lines.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
            ref_text = " ".join(words)
            r = norm(ref_text)
            if not r:
                continue
            d, target = min((best_substring(r, g) for g in gt_lines), key=lambda t: t[0])
            ok = len(target) >= MIN_LEN and d / len(r) <= MAX_LABEL_DIST
            rows.append(dict(card_id=card, box=k, x0=b[0], y0=b[1], x1=b[2], y1=b[3], ref_text=ref_text,
                             target=target if ok else "", label_dist=d / len(r), labelled=ok,
                             digits=sum(c.isdigit() for c in target) >= 4 if ok else False))
    lay = pd.DataFrame(rows)
    print(f"layout: {len(lay)} reference boxes, {lay.labelled.sum()} labelled with ground truth "
          f"({lay[lay.labelled].groupby('card_id').size().median():.0f} per card)", flush=True)
    return lay


def crop(img, b, jitter=None):
    h, w = img.shape[:2]
    dx, dy = jitter if jitter is not None else (0, 0)
    x0, y0 = int(round(b[0])) - PAD + dx, int(round(b[1])) - PAD + dy
    x1, y1 = int(round(b[2])) + PAD + dx, int(round(b[3])) + PAD + dy
    c = img[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    if c.shape[0] >= 1.5 * c.shape[1]:  # vertical text: same rule as the PaddleOCR pipeline
        c = np.ascontiguousarray(np.rot90(c))
    return c


# --------------------------------------------------------------------------
# Recognisers
# --------------------------------------------------------------------------
def make_recogniser(engine: str):
    if engine == "paddle":
        from paddleocr import TextRecognition
        model = TextRecognition(model_name="PP-OCRv6_medium_rec")

        def rec(crops):
            return [r["rec_text"] for r in model.predict(crops, batch_size=1)]
        return rec
    if engine == "tesseract":
        import pytesseract

        def rec(crops):
            out = []
            for c in crops:
                c = cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                c = cv2.copyMakeBorder(c, 10, 10, 10, 10, cv2.BORDER_REPLICATE)  # Tesseract wants a margin
                out.append(pytesseract.image_to_string(cv2.cvtColor(c, cv2.COLOR_BGR2RGB),
                                                       lang="eng", config="--psm 7").strip())
            return out
        return rec
    raise SystemExit(f"unknown engine {engine}")


def recognise_image(rec, img_path: Path, boxes: pd.DataFrame, cache_dir: Path, jitter_seed=None):
    """Readings of every box of the card on one image (cached by path + mtime + boxes)."""
    st = img_path.stat()
    sig = json.dumps([[round(v, 1) for v in b] for b in boxes[["x0", "y0", "x1", "y1"]].to_numpy().tolist()])
    key = hashlib.sha1((f"{img_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{sig}|{jitter_seed}" + (f"|pad{PAD}" if PAD else "")).encode()).hexdigest()
    f = cache_dir / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    img = cv2.imread(str(img_path))
    jit = [None] * len(boxes)
    if jitter_seed is not None:  # noise-floor proxy: tiny shift + sensor noise + JPEG
        rng = np.random.default_rng(jitter_seed)
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
        jit = [tuple(rng.integers(-2, 3, 2)) for _ in range(len(boxes))]
    crops = [crop(img, b, j) for b, j in zip(boxes[["x0", "y0", "x1", "y1"]].to_numpy(), jit)]
    texts = rec(crops)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(texts))
    return texts


# --------------------------------------------------------------------------
# Zones and scoring
# --------------------------------------------------------------------------
def zone_of(b, core, glare):
    h, w = core.shape
    x0, y0 = max(int(round(b[0])), 0), max(int(round(b[1])), 0)
    x1, y1 = min(int(round(b[2])), w), min(int(round(b[3])), h)
    if x1 <= x0 or y1 <= y0:
        return "outside"
    if core[y0:y1, x0:x1].mean() >= OVERLAP:
        return "core"
    if glare[y0:y1, x0:x1].mean() >= OVERLAP:
        return "halo"
    return "outside"


def zone_members(z: pd.DataFrame, zone: str) -> pd.DataFrame:
    if zone in ("core", "halo", "outside"):
        return z[z.zone == zone]
    if zone == "glare":
        return z[z.zone != "outside"]
    if zone == "all":
        return z
    if zone == "digits_glare":
        return z[z.digits & (z.zone != "outside")]
    if zone == "digits_all":
        return z[z.digits]
    raise ValueError(zone)


def image_scores(boxes: pd.DataFrame) -> dict:
    out = {}
    for zone in ZONES:
        g = zone_members(boxes, zone)
        out[f"n_{zone}"] = len(g)
        if len(g):
            out[f"char_acc_{zone}"] = max(0.0, 1 - g.ed.sum() / g.tlen.sum())
            out[f"exact_{zone}"] = float((g.ed == 0).mean())
        else:
            out[f"char_acc_{zone}"] = out[f"exact_{zone}"] = np.nan
    return out


def main():
    global PAD, FOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--layout", default=None, type=Path,
                    help="precomputed box layout CSV (card_id, box, x0..y1, target, labelled, digits); "
                         "default: build it from the reference OCR + ground_truth/")
    ap.add_argument("--tag", default="", help="results go to RESULTS_DIR/box_{engine}_{tag}/")
    ap.add_argument("--out-root", default="out", type=Path, help="folder holding the --runs folders")
    ap.add_argument("--pad", type=int, default=PAD, help="px of context around each box")
    ap.add_argument("--fold-accents", action="store_true")
    ap.add_argument("--engine", default="paddle", choices=["paddle", "tesseract"])
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="folders of out/ to evaluate (missing ones are skipped)")
    ap.add_argument("--extra", default="", help="more runs as dir:name,dir:name")
    ap.add_argument("--only", action="store_true", help="skip the untreated / reference / ref_jitter runs")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--gt-dir", default="ground_truth", type=Path)
    ap.add_argument("--results-dir", default="results/glare", type=Path)
    ap.add_argument("--cache-dir", default="results/ocr_cache", type=Path)
    a = ap.parse_args()
    PAD, FOLD = a.pad, a.fold_accents

    out_dir = a.results_dir / (f"box_{a.engine}_{a.tag}" if a.tag else f"box_{a.engine}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = a.cache_dir / f"box_{a.engine}"
    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs.usable & pairs.split.isin(a.splits.split(","))].reset_index(drop=True)

    if a.layout:
        lay = privacy.read_csv(a.layout, keep_default_na=False)
        lay["labelled"] = lay.labelled.astype(str) == "True"
        lay["digits"] = lay.digits.astype(str) == "True"
    else:
        lay = build_layout(pairs, a.gt_dir, a.cache_dir / "paddleocr_default")
        privacy.save_csv(lay, "results/layouts/box_layout.csv")  # real cards' text only in private/
    lay = lay[lay.labelled]

    # zones per (image, box), from the untreated image
    zones = {}
    for r in pairs.itertuples():
        un = cv2.imread(r.path)
        core = detect_core(un, DEFAULTS)
        if isinstance(getattr(r, "glare_mask_path", None), str):  # precomputed (prepare_midv.py)
            glare = cv2.imread(r.glare_mask_path, cv2.IMREAD_GRAYSCALE) > 0
        else:
            ref = cv2.imread(r.ref_path)
            glare = glare_region(cv2.cvtColor(un, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        b = lay[lay.card_id == r.card_id]
        zones[r.path] = [zone_of(x, core, glare) for x in b[["x0", "y0", "x1", "y1"]].to_numpy()]

    runs = [] if a.only else [("untreated", None), ("reference", "REF"), ("ref_jitter", "JIT")]
    runs += [(n, a.out_root / n) for n in a.runs.split(",") if n and (a.out_root / n).is_dir()]
    runs += [(n, Path(d)) for d, n in (e.split(":") for e in a.extra.split(",") if e)]

    rec = make_recogniser(a.engine)
    summaries = []
    for name, src in runs:
        t0 = time.time()
        rows, box_rows = [], []
        for r in pairs.itertuples():
            b = lay[lay.card_id == r.card_id]
            jitter = None
            if src is None:
                p = Path(r.path)
            elif src == "REF":
                p = Path(r.ref_path)
            elif src == "JIT":
                p, jitter = Path(r.ref_path), int(hashlib.md5(r.path.encode()).hexdigest()[:8], 16)
            else:
                p = src / Path(r.path).name
            if not p.exists():
                print(f"[warn] {name}: missing {p}")
                continue
            texts = recognise_image(rec, p, b, cache, jitter)
            bx = b[["box", "target", "digits"]].copy()
            bx["hyp"] = texts
            bx["zone"] = zones[r.path]
            bx["tlen"] = bx.target.str.len()
            bx["ed"] = [edit_distance(t, norm(h)) for t, h in zip(bx.target, bx.hyp)]
            bx.insert(0, "image", Path(r.path).name)
            box_rows.append(bx)
            rows.append(dict(image=Path(r.path).name, card_id=r.card_id, condition=r.condition, split=r.split,
                             **image_scores(bx)))
        res = pd.DataFrame(rows)
        res.to_csv(out_dir / f"{name}_per_image.csv", index=False)
        allb = pd.concat(box_rows)
        privacy.save_csv(allb, out_dir / f"{name}_boxes.csv")
        meta = pairs.set_index(pairs.path.map(lambda s: Path(s).name))[["split", "condition"]]
        allb = allb.join(meta, on="image")
        for (split, cond), g in allb.groupby(["split", "condition"]):
            for zone in ZONES:
                zz = zone_members(g, zone)
                if len(zz):
                    summaries.append(dict(engine=a.engine, method=name, split=split, condition=cond, zone=zone,
                                          n_boxes=len(zz), char_acc=max(0.0, 1 - zz.ed.sum() / zz.tlen.sum()),
                                          exact=(zz.ed == 0).mean()))
        print(f"{name:20s} {len(res)} images, {len(allb)} boxes  ({time.time() - t0:.0f}s)", flush=True)

    summ = pd.DataFrame(summaries)
    f = out_dir / "summary.csv"
    if a.only and f.exists():  # merge with the runs already there
        old = pd.read_csv(f)
        summ = pd.concat([old[~old.method.isin(summ.method.unique())], summ])
    summ.to_csv(f, index=False)
    for split in ("test",) if "test" in a.splits.split(",") else a.splits.split(","):
        t = summ[(summ.split == split) & summ.zone.isin(["core", "halo", "outside", "digits_glare"])]
        if len(t):
            piv = t.pivot_table(index="method", columns=["condition", "zone"], values="char_acc").round(3)
            print(f"\n=== char accuracy, {split} ({a.engine}) ===\n{piv.to_string()}")
            nb = t.pivot_table(index="method", columns=["condition", "zone"], values="n_boxes").iloc[0]
            print("boxes:", dict(nb.astype(int)))
    print(f"\nSaved {out_dir}/")


if __name__ == "__main__":
    main()
