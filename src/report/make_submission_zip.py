#!/usr/bin/env python3
"""
Submission archive: the repository with our data and generated files, without the real ID cards.

Included: everything in the working tree (code, results, report, slides, docs, data/, out/, checkpoints/), except
  * git metadata, caches, the venv, LaTeX build files, .DS_Store;
  * public downloads, re-downloadable with README "Setup": data/external/ (MIDV-2020), third_party/ (DocSHRNet),
    the pretrained Restormer weights, the ICASSP author kit (empty data/external/midv2020/, third_party/ and
    third_party/weights/ folders are kept, so the paths in README exist);
  * the 14 real cards (common/privacy.py): every file named after one (photos, outputs, overlays, masks, ground
    truth) and private/ (full-text tables).  The tracked tables with card text are already redacted; the script stops
    if a CSV still has real-card text.
A data/README.txt in the archive says what was left out.  Scores of all 22 cards stay, so make_tables.py rebuilds
every reported number from the archive.

Usage:  python src/report/make_submission_zip.py [OUT.zip]          (default ../<repo name>.zip; ~7 GB, a few minutes)
"""
import argparse
import csv
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common.privacy import TEXT_COLUMNS, is_private  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIR_NAMES = {".git", ".claude", ".pytest_cache", "__pycache__", ".ipynb_checkpoints", ".vscode", ".idea"}
SKIP_PATHS = {"private", "card_scanning_env", "third_party", "data/external", "results/ocr_cache",
              "docs/course/icassp2027_template"}
KEEP_EMPTY = ["third_party", "third_party/weights", "data/external", "data/external/midv2020"]
SKIP_FILES = {".gitattributes", ".gitignore", "checkpoints/blur/single_image_defocus_deblurring.pth"}
LATEX_BUILD = (".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz")
STORED = {".png", ".jpg", ".jpeg", ".pt", ".pth", ".pdf", ".pptx", ".gz", ".tar"}  # already compressed
CARD = re.compile(r"(?:^|_)([BF]\d\d)(?:_|\.)")
NOTE = """Data in this archive

Only the 8 printed specimen cards (B01-B03, F01-F05) are included. The 14 real ID cards (B04-B08, F06-F14) were
left out for privacy: their photos, every processed version of them and their ground-truth text. In the tables that
contain card text (results/layouts/box_layout.csv and the *_boxes.csv readings of our cards) their rows are kept with
the text left blank. All results (per-image scores, summaries, report tables) cover all 22 cards, so the reported
numbers can be rebuilt from results/ (src/report/make_tables.py), but the methods can only be re-run on the specimen
cards. MIDV-2020, DocSHRNet and the pretrained Restormer weights are public: see README "Setup" for the downloads.
"""


def names_real_card(filename):
    return any(is_private(c) for c in CARD.findall(filename))


def real_card_text(path):
    """True if a CSV has a non-empty text cell in a real card's row."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = csv.reader(f)
        head = next(rows, [])
        text = [i for i, c in enumerate(head) if c in TEXT_COLUMNS]
        keys = [i for i, c in enumerate(head) if c in ("card_id", "image")]
        if not text or not keys:
            return False
        return any(len(r) == len(head) and any(is_private(r[k][:3]) for k in keys) and any(r[i] for i in text)
                   for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path, default=ROOT.parent / f"{ROOT.name}.zip")
    out = ap.parse_args().out.resolve()
    top, n, size = ROOT.name, 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for d, dirs, files in os.walk(ROOT):
            rel_d = Path(d).relative_to(ROOT).as_posix()
            rel_d = "" if rel_d == "." else rel_d
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIR_NAMES and f"{rel_d}/{x}".lstrip("/") not in SKIP_PATHS)
            for f in sorted(files):
                rel, p = f"{rel_d}/{f}".lstrip("/"), Path(d) / f
                if (rel in SKIP_FILES or f == ".DS_Store" or p.suffix in (".pyc", ".zip") or p.is_symlink()
                        or names_real_card(f) or (rel_d == "report" and f.endswith(LATEX_BUILD))):
                    continue
                if p.suffix == ".csv" and real_card_text(p):
                    z.close()
                    out.unlink()
                    raise SystemExit(f"{rel} has real-card text: write it with common.privacy.save_csv first")
                z.write(p, f"{top}/{rel}",
                        compress_type=zipfile.ZIP_STORED if p.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
                n, size = n + 1, size + p.stat().st_size
        z.writestr(f"{top}/data/README.txt", NOTE)
        for e in KEEP_EMPTY:
            z.writestr(zipfile.ZipInfo(f"{top}/{e}/"), "")
    print(f"{n} files, {size / 1e9:.2f} GB -> {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
