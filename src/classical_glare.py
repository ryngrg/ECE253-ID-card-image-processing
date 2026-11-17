#!/usr/bin/env python3
"""
Classical glare removal for rectified ID card photos.  Two regions, as planned:

  LOST region     the clipped core of the glare (all channels ~255).  The information is
                  gone, so it is filled by inpainting (cv2.inpaint).
  PARTIAL region  the washed-out halo around the core.  Glare adds a white veil whose
                  strength falls off with distance from the core:
                      I = R * (1 - a) + 255 * a,      a = alpha0 * exp(-distance / tau)
                  so the card is recovered as R = (I - 255 a) / (1 - a), applied to the
                  lightness channel only (Lab L), which restores faded strokes without
                  adding colour noise.  Pixels with a ~ 0 are left bit-identical.

Detection needs no reference image: a core is a connected blob of clipped pixels
(min channel >= clip_thresh) with min_blob <= area <= max_core_frac of the image.

Parameters (all tunable): clip_thresh, min_blob, max_core_frac, border, core_dilate, inpaint_r,
alpha0, tau, plus use_inpaint / use_veil switches for ablations.

Usage
  python classical_glare.py --tune                      # grid search on val S_G images (maximise
                                                        # ssim_glare without lowering ssim_outside),
                                                        # writes classical_params.json
  python classical_glare.py --params classical_params.json --out-dir out/classical
  python classical_glare.py --params classical_params.json --no-inpaint --out-dir out/classical_veil
  python classical_glare.py --params classical_params.json --no-veil    --out-dir out/classical_inpaint
  python classical_glare.py ... --debug                 # also saves core outlines in OUT_DIR/debug/
  then: python evaluate.py --treated-dir out/classical --name classical

Limits: an image with no clipped core is returned unchanged (glare that never saturates is
not handled); large truly-white card areas could be mistaken for cores (check --debug and
ssim_outside); lost text cannot be recovered, only filled in plausibly.
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate import glare_region, ssim_scores

DEFAULTS = dict(
    clip_thresh=250,     # min channel >= this counts as clipped
    min_blob=100,        # px, ignore smaller clipped specks
    max_core_frac=0.25,  # ignore clipped blobs larger than this fraction of the image (white card areas)
    border=8,            # px, ignore clipped pixels this close to the image edge (card edge / corners)
    core_dilate=9,       # px, grow the core a little to cover its soft edge
    inpaint_r=5,         # cv2.inpaint radius
    alpha0=0.5,          # veil strength next to the core (0 = none)
    tau=60.0,            # px, distance over which the veil fades out
    use_inpaint=True,
    use_veil=True,
)


# --------------------------------------------------------------------------
def detect_core(img: np.ndarray, p: dict) -> np.ndarray:
    """Boolean mask of the clipped glare core(s)."""
    h, w = img.shape[:2]
    clipped = (img.min(axis=2) >= p["clip_thresh"]).astype(np.uint8)
    b = p["border"]
    if b > 0:
        clipped[:b], clipped[-b:], clipped[:, :b], clipped[:, -b:] = 0, 0, 0, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    keep = np.zeros_like(clipped)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if p["min_blob"] <= area <= p["max_core_frac"] * h * w:
            keep[lab == i] = 1
    if p["core_dilate"] > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["core_dilate"],) * 2)
        keep = cv2.dilate(keep, k)
    return keep.astype(bool)


def remove_glare(img: np.ndarray, p: dict) -> np.ndarray:
    p = {**DEFAULTS, **p}
    core = detect_core(img, p)
    if not core.any():
        return img.copy()

    out = img.copy()
    if p["use_inpaint"]:
        out = cv2.inpaint(out, core.astype(np.uint8) * 255, p["inpaint_r"], cv2.INPAINT_TELEA)

    if p["use_veil"] and p["alpha0"] > 0:
        dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = (p["alpha0"] * np.exp(-dist / p["tau"])).astype(np.float32)
        alpha[dist > 3 * p["tau"]] = 0
        zone = alpha > 0.003
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] = np.clip((lab[..., 0] - 255 * alpha) / (1 - alpha), 0, 255)
        corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        out[zone] = corrected[zone]  # everything outside the zone stays bit-identical
    return out


# --------------------------------------------------------------------------
def load_pairs(path: Path, splits=None, conditions=None) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    pairs = pairs[pairs.usable]
    if splits:
        pairs = pairs[pairs.split.isin(splits)]
    if conditions:
        pairs = pairs[pairs.condition.isin(conditions)]
    return pairs.reset_index(drop=True)


def run_batch(pairs: pd.DataFrame, p: dict, out_dir: Path, debug: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"
    if debug:
        dbg.mkdir(exist_ok=True)
    n_core = 0
    for r in pairs.itertuples():
        img = cv2.imread(r.path)
        out = remove_glare(img, p)
        cv2.imwrite(str(out_dir / Path(r.path).name), out)
        core = detect_core(img, {**DEFAULTS, **p})
        n_core += core.any()
        if debug:
            vis = img.copy()
            cnts, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            cv2.imwrite(str(dbg / Path(r.path).name), np.hstack([vis, out]))
    print(f"Processed {len(pairs)} images into {out_dir}/  (clipped core found in {n_core}, "
          f"left unchanged in {len(pairs) - n_core})")


# --------------------------------------------------------------------------
def tune(pairs_path: Path, out_json: Path, max_harm: float = 0.002):
    """Grid search on val S_G images.  Maximise ssim_glare, but only among settings that do not
    lower ssim_outside by more than max_harm (hard do-no-harm constraint): a method that repairs
    the glare by disturbing the rest of the card is not acceptable."""
    val = load_pairs(pairs_path, splits=["val"], conditions=["S_G"])
    if val.empty:
        raise SystemExit("No usable val S_G images in pairs.csv")
    data = []
    for r in val.itertuples():
        img, ref = cv2.imread(r.path), cv2.imread(r.ref_path)
        rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        data.append((img, rg, glare_region(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rg)))

    def scores(p, treat=True):
        """mean (whole, glare, outside) SSIM over the val images"""
        s = np.array([ssim_scores(rg, cv2.cvtColor(remove_glare(img, p) if treat else img, cv2.COLOR_BGR2GRAY), m)
                      for img, rg, m in data])
        return np.nanmean(s, axis=0)

    fmt = lambda sc: f"whole {sc[0]:.4f}  glare {sc[1]:.4f}  outside {sc[2]:.4f}"
    base = scores(DEFAULTS, treat=False)
    grid = dict(alpha0=[0.3, 0.5, 0.7], tau=[10, 20, 40, 60], core_dilate=[5, 9, 15], clip_thresh=[248, 250, 252])
    keys = list(grid)
    print(f"Tuning on {len(val)} val S_G images, {int(np.prod([len(v) for v in grid.values()]))} combinations")
    print(f"untreated: {fmt(base)}")
    results = [(scores({**DEFAULTS, **dict(zip(keys, c))}), dict(zip(keys, c)))
               for c in itertools.product(*grid.values())]
    ok = [r for r in results if r[0][2] >= base[2] - max_harm]
    print(f"{len(ok)}/{len(results)} settings keep ssim_outside within {max_harm} of untreated")
    if ok:
        pool = sorted(ok, key=lambda t: -t[0][1])  # best ssim_glare among the harmless ones
    else:
        print("[warn] no setting avoids harm: picking the least harmful; consider --no-veil")
        pool = sorted(results, key=lambda t: -t[0][2])
    print("\nTop 5:")
    for sc, c in pool[:5]:
        print(f"  {fmt(sc)}  {c}")
    best = {**DEFAULTS, **pool[0][1]}
    print("\nAblation on val (best params):")
    for name, extra in [("full", {}), ("inpaint only", dict(use_veil=False)),
                        ("veil only", dict(use_inpaint=False)), ("none", dict(use_inpaint=False, use_veil=False))]:
        print(f"  {name:13s} {fmt(scores({**best, **extra}))}")
    edge = [k for k, v in grid.items() if pool[0][1][k] in (min(v), max(v))]
    if edge:
        print(f"\n[note] best value of {', '.join(edge)} is at the edge of its grid: consider widening it.")
    out_json.write_text(json.dumps(best, indent=2))
    print(f"\nSaved best parameters to {out_json}")
    if len(val) < 12:
        print(f"[note] only {len(val)} val images: treat the choice as approximate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--out-dir", default="out/classical", type=Path)
    ap.add_argument("--params", type=Path, default=None, help="JSON with parameters (from --tune)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--max-harm", type=float, default=0.002,
                    help="--tune: largest allowed drop of ssim_outside vs untreated")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--no-inpaint", action="store_true")
    ap.add_argument("--no-veil", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if a.tune:
        return tune(a.pairs, a.params or Path("classical_params.json"), a.max_harm)

    p = dict(DEFAULTS)
    if a.params:
        p.update(json.loads(a.params.read_text()))
    if a.no_inpaint:
        p["use_inpaint"] = False
    if a.no_veil:
        p["use_veil"] = False
    print("Parameters:", {k: p[k] for k in sorted(p)})
    run_batch(load_pairs(a.pairs, splits=a.splits.split(",")), p, a.out_dir, a.debug)


if __name__ == "__main__":
    main()
