#!/usr/bin/env python3
"""
Prepare the glare-removal dataset.  Run AFTER rectify_ids.py (which now also aligns
every image to its card's S_D reference and writes alignment_log.csv).

Expects rectified images named  {card_id}_{B|S}_{G|D}_{num}_rectified.png

What it does
  1. Parses filenames and pairs every glare image (S_G, B_G) with its card's S_D image.
  2. Reads <rect-dir>/alignment_log.csv and checks image sizes, then marks each image
     usable / not usable (not aligned, no reference, or size/orientation mismatch).
  3. Creates a card-level train/val/test split (stratified by the card-id letter prefix,
     e.g. B01.. vs F01..) and saves it to split.csv.  An existing split.csv is never
     overwritten unless you pass --resplit.
  4. Also pairs every blur-only image (B_D) with the same S_D reference, with the same checks and the
     same split: out/pairs_blur.csv (training / tuning / testing of the deblurring methods).
  5. Saves red/cyan overlays (ref in red, image in cyan: gray = aligned, colour fringes =
     misaligned) for every flagged image, the lowest-ECC-correlation images, and a
     random sample.

Outputs
  split.csv             card_id, split                 (share with your teammate)
  out/pairs.csv         one row per glare image; use  usable == True  downstream
  out/pairs_blur.csv    one row per B_D image (same columns)
  out/overlays/*.png

Usage
  python prepare_data.py
  python prepare_data.py --rect-dir data/rectified --out-dir out --seed 0
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PATTERN = re.compile(
    r"^(?P<card_id>.+)_(?P<blur>[BS])_(?P<glare>[GD])_(?P<num>\d+)_rectified\.png$"
)
SUFFIX = "_rectified.png"


# --------------------------------------------------------------------------
# 1. Parse filenames, pair with references
# --------------------------------------------------------------------------
def build_table(rect_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(rect_dir.glob("*.png")):
        m = PATTERN.match(p.name)
        if not m:
            print(f"[skip] unrecognised filename: {p.name}")
            continue
        d = m.groupdict()
        d["num"] = int(d["num"])
        d["path"] = str(p)
        rows.append(d)
    if not rows:
        raise SystemExit(f"No matching images found in {rect_dir}")
    return pd.DataFrame(rows)


def attach_references(df: pd.DataFrame, rows: pd.Series) -> pd.DataFrame:
    """Return the images selected by `rows`, each with the card's S_D reference."""
    refs = df[(df.blur == "S") & (df.glare == "D")]
    ref_map = {}
    for card, g in refs.groupby("card_id"):
        g = g.sort_values("num")
        if len(g) > 1:
            print(f"[warn] {card}: {len(g)} reference images, using {Path(g.iloc[0].path).name}")
        ref_map[card] = g.iloc[0].path

    missing = sorted(set(df.card_id) - set(ref_map))
    if missing:
        print(f"[warn] dropping cards with no S_D reference: {missing}")

    out = df[rows].copy()
    out["ref_path"] = out.card_id.map(ref_map)
    out = out.dropna(subset=["ref_path"]).reset_index(drop=True)
    out["condition"] = out.blur + "_" + out.glare  # S_G, B_G or B_D
    return out


# --------------------------------------------------------------------------
# 2. Usability: rectify log + size check
# --------------------------------------------------------------------------
def merge_alignment_log(pairs: pd.DataFrame, rect_dir: Path) -> pd.DataFrame:
    pairs["stem"] = pairs.path.map(lambda s: Path(s).name[: -len(SUFFIX)])
    log_path = rect_dir / "alignment_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        log["stem"] = log["name"].str.replace(r"\.[^.]+$", "", regex=True)
        log = log.rename(columns={"status": "rectify_status"})
        pairs = pairs.merge(log[["stem", "rectify_status", "max_corner_shift_px", "ecc_cc"]],
                            on="stem", how="left")
        pairs["rectify_status"] = pairs.rectify_status.fillna("not in log")
    else:
        print(f"[warn] {log_path} not found: alignment status unknown, "
              f"only sizes and overlays can be checked")
        pairs["rectify_status"] = "no log"
        pairs["max_corner_shift_px"] = np.nan
        pairs["ecc_cc"] = np.nan
    return pairs


def check_sizes(pairs: pd.DataFrame) -> pd.DataFrame:
    match = []
    for _, r in pairs.iterrows():
        a, b = cv2.imread(r.path), cv2.imread(r.ref_path)
        match.append(a is not None and b is not None and a.shape == b.shape)
    pairs["size_match"] = match
    return pairs


def mark_usable(pairs: pd.DataFrame) -> pd.DataFrame:
    bad_status = pairs.rectify_status.str.startswith(("NOT-ALIGNED", "NO-REFERENCE"))
    pairs["usable"] = pairs.size_match & ~bad_status
    reason = np.where(~pairs.size_match, "size/orientation differs from reference",
                      np.where(bad_status, pairs.rectify_status, ""))
    pairs["problem"] = reason
    return pairs


# --------------------------------------------------------------------------
# 3. Split by card
# --------------------------------------------------------------------------
def make_split(cards, seed: int, val_frac: float, test_frac: float, stratify: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups: dict[str, list] = {}
    for c in sorted(cards):
        key = re.match(r"[A-Za-z]*", c).group(0) if stratify else ""
        groups.setdefault(key, []).append(c)
    rows = []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(test_frac * n)) if n >= 3 else 0
        n_val = max(1, round(val_frac * n)) if n >= 3 else 0
        for i, c in enumerate(g):
            rows.append((c, "test" if i < n_test else "val" if i < n_test + n_val else "train"))
    return pd.DataFrame(rows, columns=["card_id", "split"])


def load_or_make_split(path: Path, cards, a) -> pd.DataFrame:
    if path.exists() and not a.resplit:
        split = pd.read_csv(path)
        new = sorted(set(cards) - set(split.card_id))
        if new:
            print(f"[warn] cards missing from existing {path}: {new} (they get no split)")
        print(f"Using existing split file: {path}")
        return split
    split = make_split(cards, a.seed, a.val_frac, a.test_frac, not a.no_stratify)
    split.to_csv(path, index=False)
    print(f"Wrote new split file: {path}")
    return split


# --------------------------------------------------------------------------
# 4. Overlays
# --------------------------------------------------------------------------
def save_overlay(row, out_path: Path):
    ref, img = cv2.imread(row.ref_path), cv2.imread(row.path)
    if ref is None or img is None:
        return
    if img.shape[:2] != ref.shape[:2]:  # display only
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_path), cv2.merge([g, g, r]))  # BGR: B=img, G=img, R=ref


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rect-dir", default="data/rectified", type=Path)
    ap.add_argument("--out-dir", default="out", type=Path)
    ap.add_argument("--split-file", default="split.csv", type=Path)
    ap.add_argument("--resplit", action="store_true", help="overwrite existing split file")
    ap.add_argument("--no-stratify", action="store_true",
                    help="plain random card split instead of stratifying by card-id letter prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--test-frac", type=float, default=0.18)
    ap.add_argument("--n-sample", type=int, default=6, help="random overlays of usable images")
    ap.add_argument("--n-lowcc", type=int, default=6, help="overlays of lowest-ECC-cc usable images")
    a = ap.parse_args()

    overlay_dir = a.out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for old in overlay_dir.glob("*.png"):
        old.unlink()

    # --- parse, pair, validate ---
    df = build_table(a.rect_dir)
    def checked(rows):
        return mark_usable(check_sizes(merge_alignment_log(attach_references(df, rows), a.rect_dir)))

    pairs = checked(df.glare == "G")
    blur = checked((df.blur == "B") & (df.glare == "D"))

    # --- split ---
    split = load_or_make_split(a.split_file, pairs.card_id.unique(), a)
    pairs = pairs.merge(split, on="card_id", how="left").drop(columns="stem")
    pairs.to_csv(a.out_dir / "pairs.csv", index=False)
    blur = blur.merge(split, on="card_id", how="left").drop(columns="stem")
    blur.to_csv(a.out_dir / "pairs_blur.csv", index=False)

    # --- overlays ---
    flagged = pairs[~pairs.usable]
    for _, r in flagged.iterrows():
        save_overlay(r, overlay_dir / f"flagged_{Path(r.path).stem}.png")
    ok = pairs[pairs.usable]
    if ok.ecc_cc.notna().any():
        for _, r in ok.nsmallest(a.n_lowcc, "ecc_cc").iterrows():
            save_overlay(r, overlay_dir / f"lowcc_{Path(r.path).stem}.png")
    n = min(a.n_sample, len(ok))
    for i in np.random.default_rng(a.seed).choice(len(ok), size=n, replace=False):
        r = ok.iloc[i]
        save_overlay(r, overlay_dir / f"sample_{Path(r.path).stem}.png")

    for _, r in blur.iterrows():  # few images: save them all
        save_overlay(r, overlay_dir / f"blur_{Path(r.path).stem}.png")

    # --- summary ---
    print(f"\n{len(df)} images parsed, {pairs.card_id.nunique()} cards with a reference, "
          f"{len(pairs)} glare images, {int(pairs.usable.sum())} usable")
    print("\nUsable glare images per condition and split:")
    print(pd.crosstab(ok.split, ok.condition).to_string())
    print("\nUsable blur-only (B_D) images per split:")
    print(blur[blur.usable].split.value_counts().to_string())
    print("\nCards per split:")
    print(split.split.value_counts().to_string())
    if len(flagged):
        print(f"\n{len(flagged)} glare images NOT usable:")
        for _, r in flagged.iterrows():
            print(f"  {Path(r.path).name:40s} {r.problem}")
    print(f"\nCheck {overlay_dir}/ (gray = aligned, red/cyan fringes = misaligned).")
    print(f"Table saved to {a.out_dir / 'pairs.csv'}")


if __name__ == "__main__":
    main()
