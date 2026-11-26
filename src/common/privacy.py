"""
Real ID cards stay private: their text is never written to a tracked file.

14 of our 22 card sides are real cards (B04-B08, F06-F14); only the 8 printed specimens (B01-B03, F01-F05) may be
shown or shared.  The images are git-ignored anyway (data/, out/).  Text read from a card is in three kinds of
files: ground_truth/{card}.txt (the real cards' files are git-ignored), results/layouts/box_layout.csv (ref_text,
target) and the box_ocr.py readings {run}_boxes.csv (target, hyp).  For those tables:

  save_csv(df, path)   writes the full table to private/<path> (git-ignored, local only) and `path` with the text
                       columns of real-card rows blanked (all numbers kept, so every score stays reproducible
                       from the tracked tables)
  read_csv(path)       reads private/<path> if it exists (full text, on the machine that has the private data),
                       else `path`

MIDV-2020 documents (card ids like alb_id_60) are public mock documents and are never redacted.
"""
import re
from pathlib import Path

import pandas as pd

SPECIMEN_CARDS = ("B01", "B02", "B03", "F01", "F02", "F03", "F04", "F05")
TEXT_COLUMNS = ("ref_text", "target", "hyp")
ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = Path("private")
_OUR_CARD = re.compile(r"^[BF]\d\d$")


def is_private(card_id: str) -> bool:
    """True for our real cards (B04-B08, F06-F14); False for the specimens and for MIDV documents."""
    return bool(_OUR_CARD.match(card_id)) and card_id not in SPECIMEN_CARDS


def card_of(df: pd.DataFrame) -> pd.Series:
    """Card id of every row: the card_id column, else the first 3 characters of the image name."""
    return df["card_id"].astype(str) if "card_id" in df else df["image"].astype(str).str[:3]


def redact(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with the text columns of real-card rows blanked."""
    out = df.copy()
    private = card_of(out).map(is_private)
    for c in TEXT_COLUMNS:
        if c in out:
            out[c] = out[c].astype(object).where(~private, "")
    return out


def private_path(path) -> Path:
    """private/<path> (relative paths are taken from the repository root; a path outside the repository goes to
    private/outside/<its absolute path>)."""
    path = Path(path)
    if not path.is_absolute():
        return ROOT / PRIVATE_ROOT / path
    path = path.resolve()
    if path.is_relative_to(ROOT):
        return ROOT / PRIVATE_ROOT / path.relative_to(ROOT)
    return ROOT / PRIVATE_ROOT / "outside" / path.relative_to(path.anchor)


def save_csv(df: pd.DataFrame, path) -> None:
    path = Path(path)
    if card_of(df).map(is_private).any():
        full = private_path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full, index=False)
        df = redact(df)
    df.to_csv(path, index=False)


def read_csv(path, **kw) -> pd.DataFrame:
    full = private_path(path)
    return pd.read_csv(full if full.exists() else path, **kw)
