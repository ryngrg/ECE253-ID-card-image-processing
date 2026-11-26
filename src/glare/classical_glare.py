#!/usr/bin/env python3
"""
Classical glare (specular highlight) removal for rectified ID card photos: a two-zone method.

Physics.  Glare from the laminate is a white specular reflection added on top of the card
(dichromatic model, white illuminant).  On our data it is well described by a white veil
        I = R * (1 - a) + 255 * a                 (per channel; a = veil strength)
In textured, unclipped glare blocks a linear fit of I against the reference R gives
gain + offset/255 = 0.96 (median), i.e. the model holds; the veil typically lifts the black
level of the ink from ~20 to ~110 (a ~ 0.35-0.55).  The text is still there, only washed out.

Two zones
  CORE  clipped pixels (min channel >= clip_thresh): the information is gone.  They are filled
        from the already corrected surroundings, either with the median colour of a ring around
        the core (core_fill=flat, final choice) or by fast-marching inpainting (core_fill=telea,
        as in the proposal).  On our OCR tests Telea smears neighbouring strokes into the hole and
        PaddleOCR misreads the smear (word F1 worse on 7 of 39 train+val images, better on 1);
        the flat fill does not invent structure.
  HALO  unclipped glare: the veil is removed by inverting the model, R = (I - 255 a) / (1 - a),
        which keeps every stroke that survived.  Nothing is synthesised there.

Estimating the veil a(x) without the reference ("statistics inpainting", veil_mode=stat)
  1. seeds: connected clipped blobs with min_blob <= area <= max_core_frac * image (if none, the
     image is returned unchanged: we only act where glare is certain)
  2. on a coarse grid (block px) measure, in a win x win window, the black level bl (q_lo-th
     percentile of the min channel; dark ink) and the contrast (q_hi - q_lo percentile).
     Blocks with contrast >= con_min contain text/graphics and carry evidence about the veil
  3. expected black level b0(x): Gaussian normalised convolution of bl over textured blocks
     FAR from the seeds (> r_max), with the global median as fallback.  This inpaints a smooth
     statistic of the card, not its pixels
  4. veil evidence in textured blocks near the seeds: a = (bl - b0) / (255 - b0), clipped to
     [0, a_max]; spread to untextured blocks by normalised convolution (sigma_a blocks).  No
     evidence -> a = 0 (do no harm)
  5. white-level cap: a <= (white - w0) / (255 - w0), w0 = expected white level from far blocks, so
     the estimate cannot leak into blank paper where the veil is weaker (it would darken it)
  6. hysteresis: keep only the veil region (a > a_min) connected to a seed, feather it, scale by
     strength, upsample (bicubic)
  7. background floor (per pixel): a pixel on the paper side of the local black/white midpoint is
     not darkened below the expected paper white (removes dark rings around cores on plain cards)
  8. invert the model on all three channels (restores colour saturation too)
  The older model (veil_mode=decay: a = alpha0 * exp(-distance_to_core / tau), on Lab L only) is
  kept for the ablation "with vs without our modification".

Parameter choice (params/glare/classical_params.json).  Design choices (flat core fill, white cap, background
floor) were made from PaddleOCR and visual checks on train+val S_G; the
numeric parameters come from --tune (best ssim_glare with ssim_outside harm <= 0.002 on train+val
S_G), and were checked to be OCR-neutral on train+val.  The test split was not used.  Note that
SSIM is a weak proxy for legibility here: smoothing variants raise SSIM most but not OCR.
Tried on train+val and dropped (no OCR gain): CLAHE in the halo, a grayscale opening against laminate
scratches, and an edge-preserving denoise before the veil inversion (NLM / guided / bilateral).

Usage
  python src/glare/classical_glare.py --tune                      # grid search on train+val S_G (maximise
                                                        # ssim_glare with ssim_outside harm <= --max-harm)
  python src/glare/classical_glare.py --params params/glare/classical_params.json --out-dir out/classical
  python src/glare/classical_glare.py --params params/glare/classical_params.json --no-veil   --out-dir out/classical_inpaint
  python src/glare/classical_glare.py --params params/glare/classical_params.json --no-inpaint --out-dir out/classical_veil
  python src/glare/classical_glare.py --params params/glare/classical_params.json --core-fill telea --out-dir out/classical_telea
  python src/glare/classical_glare.py --veil-mode decay --params params/glare/classical_params_decay.json --out-dir out/classical_decay
  add --debug to save [input | core+veil map | output] panels in OUT_DIR/debug/
  then: python src/common/evaluate.py --engine paddleocr --treated-dir out/classical --name classical

Limits: glare that never clips is left alone; text inside a clipped core is not recovered
(inpainting only makes the hole unobtrusive); a light-coloured graphic next to a seed can look
like veil (bounded by a_max and the do-no-harm tuning).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common.evaluate import glare_region, ssim_scores

DEFAULTS = dict(
    # seeds / core
    clip_thresh=248,     # min channel >= this counts as clipped
    min_blob=100,        # px, ignore smaller clipped specks
    max_core_frac=0.25,  # ignore clipped blobs larger than this fraction of the image (white areas)
    border=8,            # px, ignore clipped pixels this close to the image edge
    core_dilate=9,       # px, grow the core to cover its soft edge before inpainting
    inpaint_r=5,         # cv2.inpaint radius
    # halo veil, statistics-inpainting model (veil_mode=stat)
    veil_mode="stat",
    block=16,            # px, grid step for the local statistics
    win=32,              # px, window for the local statistics
    q_lo=3.0,            # percentile used as the local black (ink) level
    q_hi=97.0,           # percentile used as the local white level
    con_min=40.0,        # min local contrast (q_hi - q_lo) for a block to carry veil evidence
    r_max=200,           # px, veil may extend this far from a seed; b0 is estimated beyond it
    sigma_ref=8.0,       # blocks, smoothing of the expected black level b0
    sigma_a=2.0,         # blocks, spreading/smoothing of the veil estimate
    a_max=0.7,           # cap on the veil (limits noise amplification 1/(1-a))
    a_min=0.05,          # hysteresis threshold for the veil region
    strength=1.0,        # scale on the estimated veil (<1 = conservative)
    white_cap=True,      # cap a by the lift of the local white level (see estimate_veil)
    cap_slack=0.05,      # tolerance added to that cap
    bg_floor=True,       # per pixel: paper-side pixels are not darkened below the expected white
    core_fill="flat",    # flat (median colour of a ring around the core) | telea (fast-marching inpainting)
    # halo veil, old distance-decay model (veil_mode=decay)
    alpha0=0.5,
    tau=60.0,
    # switches
    use_inpaint=True,
    use_veil=True,
)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_core(img: np.ndarray, p: dict, dilate: bool = True) -> np.ndarray:
    """Boolean mask of the clipped glare core(s) (the seeds)."""
    h, w = img.shape[:2]
    clipped = (img.min(axis=2) >= p["clip_thresh"]).astype(np.uint8)
    b = p["border"]
    if b > 0:
        clipped[:b], clipped[-b:], clipped[:, :b], clipped[:, -b:] = 0, 0, 0, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    ok = (areas >= p["min_blob"]) & (areas <= p["max_core_frac"] * h * w)
    ok[0] = False
    keep = ok[lab].astype(np.uint8)
    if dilate and p["core_dilate"] > 1 and keep.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["core_dilate"],) * 2)
        keep = cv2.dilate(keep, k)
    return keep.astype(bool)


def _grid_stats(mn: np.ndarray, block: int, win: int, q_lo: float, q_hi: float):
    """Percentiles of `mn` in win x win windows centred on a grid with step `block`.
    Returns (black, white) maps of shape (ceil(H/block), ceil(W/block))."""
    h, w = mn.shape
    gh, gw = -(-h // block), -(-w // block)
    pad = (win - block) // 2
    padded = np.pad(mn, ((pad, gh * block - h + pad), (pad, gw * block - w + pad)), mode="reflect")
    v = sliding_window_view(padded, (win, win))[::block, ::block][:gh, :gw]
    flat = v.reshape(gh, gw, -1)
    lo, hi = np.percentile(flat, [q_lo, q_hi], axis=2)
    return lo.astype(np.float32), hi.astype(np.float32)


def _norm_conv(values, weights, sigma, fallback=None, min_w=1e-3):
    """Gaussian normalised convolution: smooth interpolation of sparse weighted samples."""
    k = int(4 * sigma) | 1
    num = cv2.GaussianBlur(values * weights, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(weights.astype(np.float32), (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    out = num / np.maximum(den, 1e-6)
    if fallback is not None:
        t = np.clip(den / min_w, 0, 1)  # blend to the fallback where there is little support
        out = t * out + (1 - t) * fallback
    else:
        out[den < min_w] = 0
    return out, den


def estimate_veil(img: np.ndarray, core: np.ndarray, p: dict) -> np.ndarray:
    """Veil strength a(x) in [0, a_max] at full resolution (float32)."""
    h, w = img.shape[:2]
    s = p["block"]
    if p["veil_mode"] == "decay":
        dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        a = (p["alpha0"] * np.exp(-dist / p["tau"])).astype(np.float32)
        a[dist > 3 * p["tau"]] = 0
        return a

    mn = img.min(axis=2).astype(np.float32)
    bl, wh = _grid_stats(mn, s, p["win"], p["q_lo"], p["q_hi"])
    gh, gw = bl.shape
    textured = (wh - bl) >= p["con_min"]

    # distance (px) of each block centre from the clipped seeds
    core_small = cv2.resize(core.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_AREA) > 0
    if not core_small.any():
        core_small = cv2.resize(core.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_LINEAR) > 0
    dist = cv2.distanceTransform((~core_small).astype(np.uint8), cv2.DIST_L2, 5) * s
    near = dist <= p["r_max"]

    # expected (glare-free) black level, from textured blocks far from any seed
    ref_w = (textured & ~near).astype(np.float32)
    glob = float(np.median(bl[ref_w > 0])) if ref_w.sum() >= 5 else float(np.percentile(bl, 10))
    b0, _ = _norm_conv(bl, ref_w, p["sigma_ref"], fallback=np.full_like(bl, glob), min_w=0.05)
    b0 = np.minimum(b0, glob + 40)  # a local estimate never far above the card's ink level

    # veil evidence in textured blocks near the seeds
    ev = np.clip((bl - b0) / np.maximum(255 - b0, 1), 0, p["a_max"]).astype(np.float32)
    ev_w = (textured & near & ~core_small).astype(np.float32)
    a, _ = _norm_conv(ev, ev_w, p["sigma_a"], fallback=None, min_w=0.05)
    a[~near] = 0
    a = np.clip(a, 0, p["a_max"])

    # white-level cap: the veil cannot be stronger than what lifts the expected white level w0
    # (from blocks far from the seeds) to the observed one.  Stops the estimate from leaking into
    # blank paper, where it would darken the background.
    far_w = (~near).astype(np.float32)
    wglob = float(np.median(wh[~near])) if (~near).sum() >= 5 else float(np.median(wh))
    w0, _ = _norm_conv(wh, far_w, p["sigma_ref"], fallback=np.full_like(wh, wglob), min_w=0.05)
    if p["white_cap"]:
        cap = np.clip((wh - w0) / np.maximum(255 - w0, 1), 0, 1) + p["cap_slack"]
        a = np.minimum(a, cv2.GaussianBlur(cap, (3, 3), 0.8))

    # hysteresis: keep the veil region connected to a seed
    region = (a > p["a_min"]).astype(np.uint8) | core_small.astype(np.uint8)
    n, lab = cv2.connectedComponents(region, connectivity=8)
    seeded = np.unique(lab[core_small])
    keep = np.isin(lab, seeded[seeded > 0])
    a = np.where(keep, a, 0).astype(np.float32)
    a = cv2.GaussianBlur(a, (5, 5), 1.0)  # feather the region edge

    # upsample (block centres -> pixels)
    up = lambda z: cv2.resize(z, (gw * s, gh * s), interpolation=cv2.INTER_CUBIC)[:h, :w]
    big = np.clip(up(a) * p["strength"], 0, p["a_max"]).astype(np.float32)

    # background floor (per pixel): a pixel on the paper side of the local black/white midpoint
    # must not come out darker than the expected paper white w0.  Caps the veil there at
    # a = (I - w0) / (255 - w0).  Removes dark rings around cores on plain backgrounds; text
    # pixels (dark side) are unaffected.
    if p["bg_floor"]:
        w0f, blf, whf = up(w0), up(bl), up(wh)
        mid = 0.5 * (blf + whf)
        bgw = np.clip((mn - mid) / np.maximum(0.25 * (whf - blf), 4.0), 0, 1)
        cap_px = np.clip((mn - w0f) / np.maximum(255 - w0f, 1), 0, 1)
        big = big - bgw * np.maximum(big - cap_px, 0)
    return big.astype(np.float32)


# --------------------------------------------------------------------------
# Removal
# --------------------------------------------------------------------------
def remove_glare(img: np.ndarray, p: dict, return_maps: bool = False):
    p = {**DEFAULTS, **p}
    core = detect_core(img, p)
    if not core.any():
        return (img.copy(), core, np.zeros(img.shape[:2], np.float32)) if return_maps else img.copy()

    out = img.copy()
    a = np.zeros(img.shape[:2], np.float32)
    if p["use_veil"]:
        a = estimate_veil(img, core, p)
        zone = a > 0.003
        if p["veil_mode"] == "decay":  # old model: lightness only
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[..., 0] = np.clip((lab[..., 0] - 255 * a) / (1 - a), 0, 255)
            corr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        else:  # white specular veil: all channels
            a3 = a[..., None]
            corr = np.clip((out.astype(np.float32) - 255 * a3) / (1 - a3) + 0.5, 0, 255).astype(np.uint8)
        out[zone] = corr[zone]  # pixels with no veil stay bit-identical

    if p["use_inpaint"]:
        if p["core_fill"] == "flat":
            # no synthesis: each core blob gets the median colour of a ring around it
            filled = out.copy()
            n, lab_ = cv2.connectedComponents(core.astype(np.uint8))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            for i in range(1, n):
                blob = (lab_ == i).astype(np.uint8)
                ring = (cv2.dilate(blob, k) > 0) & ~core
                if ring.any():
                    filled[blob > 0] = np.median(out[ring], axis=0).astype(np.uint8)
        else:
            filled = cv2.inpaint(out, core.astype(np.uint8) * 255, p["inpaint_r"], cv2.INPAINT_TELEA)
        # feathered blend at the core boundary to avoid a hard seam
        wgt = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 2.0)
        wgt = np.maximum(wgt, core)[..., None]
        out = (wgt * filled + (1 - wgt) * out + 0.5).astype(np.uint8)
    return (out, core, a) if return_maps else out


# --------------------------------------------------------------------------
# Batch / tuning
# --------------------------------------------------------------------------
def load_pairs(path: Path, splits=None, conditions=None) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    pairs = pairs[pairs.usable]
    if splits:
        pairs = pairs[pairs.split.isin(splits)]
    if conditions:
        pairs = pairs[pairs.condition.isin(conditions)]
    return pairs.reset_index(drop=True)


def debug_panel(img, out, core, a):
    vis = img.copy()
    heat = cv2.applyColorMap((np.clip(a / 0.7, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    vis = cv2.addWeighted(vis, 0.5, heat, 0.5, 0)
    cnts, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return np.hstack([img, vis, out])


def run_batch(pairs: pd.DataFrame, p: dict, out_dir: Path, debug: bool, in_dir: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"
    if debug:
        dbg.mkdir(exist_ok=True)
    n_core, t0 = 0, time.time()
    for r in pairs.itertuples():
        src = in_dir / Path(r.path).name if in_dir else Path(r.path)
        img = cv2.imread(str(src))
        if img is None:
            raise SystemExit(f"cannot read {src}")
        out, core, a = remove_glare(img, p, return_maps=True)
        cv2.imwrite(str(out_dir / Path(r.path).name), out)
        n_core += core.any()
        if debug:
            cv2.imwrite(str(dbg / Path(r.path).name), debug_panel(img, out, core, a))
    print(f"Processed {len(pairs)} images into {out_dir}/ in {time.time() - t0:.1f}s  "
          f"(seeds found in {n_core}, left unchanged in {len(pairs) - n_core})")


GRIDS = {
    "stat": dict(strength=[0.5, 0.75, 1.0], a_max=[0.6, 0.85], r_max=[100, 200], a_min=[0.05, 0.15],
                 cap_slack=[0.05, 0.15], con_min=[30, 50]),
    "decay": dict(alpha0=[0.3, 0.5, 0.7], tau=[10, 20, 40, 60], core_dilate=[5, 9, 15]),
}


def tune(pairs_path: Path, out_json: Path, mode: str, splits, max_harm: float = 0.002):
    """Grid search on S_G images of `splits`.  Maximise ssim_glare among the settings that do not
    lower ssim_outside by more than max_harm (do no harm)."""
    val = load_pairs(pairs_path, splits=splits, conditions=["S_G"])
    if val.empty:
        raise SystemExit("No usable S_G images to tune on")
    data = []
    for r in val.itertuples():
        img, ref = cv2.imread(r.path), cv2.imread(r.ref_path)
        rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        data.append((img, rg, glare_region(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rg)))

    def scores(p, treat=True):
        s = np.array([ssim_scores(rg, cv2.cvtColor(remove_glare(img, p) if treat else img, cv2.COLOR_BGR2GRAY), m)
                      for img, rg, m in data])
        return np.nanmean(s, axis=0)

    fmt = lambda sc: f"whole {sc[0]:.4f}  glare {sc[1]:.4f}  outside {sc[2]:.4f}"
    base_p = {**DEFAULTS, "veil_mode": mode}
    base = scores(base_p, treat=False)
    grid = GRIDS[mode]
    keys = list(grid)
    n = int(np.prod([len(v) for v in grid.values()]))
    print(f"Tuning veil_mode={mode} on {len(val)} S_G images ({','.join(splits)}), {n} combinations")
    print(f"untreated: {fmt(base)}")
    t0, results = time.time(), []
    for i, c in enumerate(itertools.product(*grid.values()), 1):
        results.append((scores({**base_p, **dict(zip(keys, c))}), dict(zip(keys, c))))
        if i % 20 == 0:
            print(f"  {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    ok = [r for r in results if r[0][2] >= base[2] - max_harm]
    print(f"{len(ok)}/{len(results)} settings keep ssim_outside within {max_harm} of untreated")
    pool = sorted(ok, key=lambda t: -t[0][1]) if ok else sorted(results, key=lambda t: -t[0][2])
    print("\nTop 5:")
    for sc, c in pool[:5]:
        print(f"  {fmt(sc)}  {c}")
    best = {**base_p, **pool[0][1]}
    print("\nAblation (best params):")
    for name, extra in [("full", {}), ("inpaint only", dict(use_veil=False)),
                        ("veil only", dict(use_inpaint=False)), ("none", dict(use_inpaint=False, use_veil=False))]:
        print(f"  {name:13s} {fmt(scores({**best, **extra}))}")
    edge = [k for k, v in grid.items() if len(v) > 1 and pool[0][1][k] in (min(v), max(v))]
    if edge:
        print(f"\n[note] best value of {', '.join(edge)} is at the edge of its grid.")
    out_json.write_text(json.dumps(best, indent=2))
    print(f"\nSaved best parameters to {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--in-dir", type=Path, default=None,
                    help="read the images from this folder (same filenames), e.g. the output of deblur.py")
    ap.add_argument("--out-dir", default="out/classical", type=Path)
    ap.add_argument("--params", type=Path, default=None, help="JSON with parameters (from --tune)")
    ap.add_argument("--veil-mode", choices=["stat", "decay"], default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-splits", default="train,val")
    ap.add_argument("--max-harm", type=float, default=0.002,
                    help="--tune: largest allowed drop of ssim_outside vs untreated")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--conditions", default="S_G,B_G")
    ap.add_argument("--no-inpaint", action="store_true")
    ap.add_argument("--no-veil", action="store_true")
    ap.add_argument("--core-fill", choices=["flat", "telea"], default=None)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if a.tune:
        return tune(a.pairs, a.params or Path("params/glare/classical_params.json"), a.veil_mode or "stat",
                    a.tune_splits.split(","), a.max_harm)

    p = dict(DEFAULTS)
    if a.params:
        p.update(json.loads(a.params.read_text()))
    if a.veil_mode:
        p["veil_mode"] = a.veil_mode
    if a.no_inpaint:
        p["use_inpaint"] = False
    if a.no_veil:
        p["use_veil"] = False
    if a.core_fill:
        p["core_fill"] = a.core_fill
    print("Parameters:", {k: p[k] for k in sorted(p)})
    run_batch(load_pairs(a.pairs, splits=a.splits.split(","), conditions=a.conditions.split(",")),
              p, a.out_dir, a.debug, a.in_dir)


if __name__ == "__main__":
    main()
