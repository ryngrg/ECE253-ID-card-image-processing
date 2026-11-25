#!/usr/bin/env python3
"""
Figures of the report and slides -> report/figures/{name}.pdf and .png (300 dpi), at the size of the ICASSP page
(column 3.39 in, full width 7 in).  Only specimen cards (F01-F05, B01-B03) and MIDV-2020 documents are shown.

  dataset          raw photos of one card: sharp / blurred x diffuse / glare (two of the light setups)
  blur_crops       text lines of test B_D images per deblur method, with the readings of both OCR engines
  glare_crops      glare regions per deglare method (zones marked on the untreated image), with readings
  chain_crops      B_G regions: single steps and both orders, with readings
  fusion           MIDV-2020 clip: key frame, another frame, best frame per field, fused, scan
  halo_changes     paired change of halo accuracy per method, engine and data set (dots, 95% CI)
  joint_changes    the same for all boxes of B_G test and MIDV photos (deblur, deglare, both orders)
  pipeline, glare_methods   block diagrams (TikZ in report/figures/*.tex, compiled with latexmk)

Readings come from results/*/box_*/*_boxes.csv; a check mark = every letter and digit correct.
make_tables.py must run first (halo_changes and joint_changes read report/tables/paired_changes.csv).

Usage:  python src/report/make_figures.py [--only blur_crops,fusion]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common.evaluate import glare_region
from glare.classical_glare import DEFAULTS, detect_core

OUT = Path("report/figures")
COL, FULL = 3.39, 7.0           # inches
SPECIMEN = {"F01", "F02", "F03", "F04", "F05", "B01", "B02", "B03"}
INK, MUTED, GRID = "#191919", "#666666", "#dddddd"
ENGINE = {"paddle": ("PaddleOCR", "#257de0"), "tesseract": ("Tesseract", "#f47230")}
CORE, HALO = (40, 110, 235), (255, 140, 0)  # RGB, zone outlines
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf, .png")


def load(path):
    img = cv2.imread(str(path))
    assert img is not None, path
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ours(name, folder="data/rectified"):
    assert name.split("_")[0] in SPECIMEN, f"{name}: not a specimen card"
    return load(f"{folder}/{name}_rectified.png")


def crop(img, b, pad=0):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    return img[max(y0 - pad, 0):min(y1 + pad, h), max(x0 - pad, 0):min(x1 + pad, w)]


def reading(folder, run, image, box):
    """Two lines of text: PaddleOCR and Tesseract reading of one box ('✓' in front = correct)."""
    out = []
    for e, tag in (("paddle", "P"), ("tesseract", "T")):
        t = pd.read_csv(f"results/{folder.format(e)}/{run}_boxes.csv", keep_default_na=False)
        r = t[(t.image == image) & (t.box == box)].iloc[0]
        hyp = str(r.hyp).strip() or "(nothing)"
        hyp = hyp if len(hyp) <= 28 else hyp[:27] + "…"
        out.append((f"{tag}: ", hyp, int(r.ed) == 0))
    return out


def put_reading(ax, rd, size=5.8):
    for k, (tag, hyp, ok) in enumerate(rd):
        s = f"{tag}{hyp}".replace("$", r"\$")
        ax.text(0.0, -0.03 - 0.0, "\n" * k + s + (" ✓" if ok else ""), transform=ax.transAxes, va="top", ha="left", fontsize=size,
                color=INK if ok else MUTED, fontfamily=["Arial", "DejaVu Sans"], linespacing=1.15, clip_on=False)


def panel_grid(name, rows, cols, width, text_lines=2, head=0.17, gap=0.05, text_pt=5.8):
    """rows: list of (list of RGB crops, list of readings or None); all crops of a row share one aspect ratio."""
    n = len(cols)
    w = (width - gap * (n - 1)) / n
    hs = [w * r[0][-1].shape[0] / r[0][-1].shape[1] for r in rows]
    th = text_lines * text_pt * 1.15 / 72 + 0.055
    H = head + sum(hs) + th * len(rows)
    fig = plt.figure(figsize=(width, H))
    y = H - head
    for (imgs, texts), h in zip(rows, hs):
        for j, img in enumerate(imgs):
            ax = fig.add_axes([j * (w + gap) / width, (y - h) / H, w / width, h / H])
            ax.imshow(img, interpolation="lanczos", aspect="auto")
            ax.set_axis_off()
            if y == H - head:
                ax.set_title(cols[j], fontsize=7, pad=2.5, color=INK)
            if texts:
                put_reading(ax, texts[j], text_pt)
        y -= h + th
    save(fig, name)


def zones(img, glare):
    core = detect_core(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), DEFAULTS).astype(np.uint8)
    out = img.copy()
    for m, c in (((glare | (core > 0)).astype(np.uint8), HALO), (core, CORE)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        m = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 300]).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, c, 3)
    return out


def our_glare(name):
    g = [cv2.cvtColor(ours(n), cv2.COLOR_RGB2GRAY) for n in (name, name[:3] + "_S_D_0")]
    return glare_region(*g)


# ------------------------------------------------------------------ figures
def dataset():
    names = [["F01_S_D_0", "F01_S_G_0", "F01_S_G_1"], ["F01_B_D_0", "F01_B_G_0", "F01_B_G_1"]]
    w, gap = (COL - 0.17 - 0.1) / 3, 0.05
    h = w * 0.62
    H = 0.16 + 2 * h + gap
    fig = plt.figure(figsize=(COL, H))
    for i, row in enumerate(names):
        for j, n in enumerate(row):
            img = load(f"data/raw/{n}.png")
            hh, ww = img.shape[:2]
            x, y, cw = int(.08 * ww), int(.14 * hh + .04 * ww), int(.78 * ww)
            img = cv2.resize(img[y:y + int(.62 * cw), x:x + cw], (900, 558), interpolation=cv2.INTER_AREA)
            ax = fig.add_axes([(0.17 + j * (w + gap)) / COL, 1 - (0.16 + i * (h + gap) + h) / H, w / COL, h / H])
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0 and j == 0:
                ax.set_title("diffuse", fontsize=7, pad=2.5)
            if i == 0 and j == 1:  # one title over both glare columns
                ax.text(1 + gap / w / 2, 1.03, "glare, different light setups", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
            if j == 0:
                ax.set_ylabel(["sharp", "blurred"][i], fontsize=7, labelpad=3)
    save(fig, "dataset")


def blur_crops():
    lay = pd.read_csv("results/layouts/box_layout.csv").set_index(["card_id", "box"])
    runs = [("untreated", None, "untreated"), ("wiener_ad", "out/blur/wiener_ad", "Wiener, adaptive"),
            ("restormer_pre", "out/blur/restormer_pre", "Restormer, pretrained"),
            ("restormer", "out/blur/restormer", "Restormer, fine-tuned"), ("reference", "REF", "reference")]
    rows = []
    for name, box in (("F04_B_D_0", 8), ("F03_B_D_0", 15), ("B03_B_D_0", 9), ("F04_B_D_0", 12)):
        b = lay.loc[(name[:3], box), ["x0", "y0", "x1", "y1"]].to_numpy(float)
        # widen every box to the same aspect ratio so that the rows have equal heights
        w = max(b[2] - b[0], 6.2 * (b[3] - b[1]))
        c = min(max((b[0] + b[2]) / 2, w / 2), 1012 - w / 2)
        b = [c - w / 2, b[1] - 3, c + w / 2, b[3] + 3]
        imgs, txt = [], []
        for run, d, _ in runs:
            img = ours(name[:3] + "_S_D_0") if d == "REF" else ours(name, d or "data/rectified")
            imgs.append(crop(img, b))
            txt.append(reading("blur/box_{}_blur", run, f"{name}_rectified.png", box))
        rows.append((imgs, txt))
    panel_grid("blur_crops", rows, [r[2] for r in runs], FULL)


def region_rows(spec, runs, folder):
    rows = []
    for name, box, b in spec:
        imgs, txt = [], []
        for run, d, _ in runs:
            if d == "ZONES":
                img = zones(ours(name), our_glare(name))
            else:
                img = ours(name[:3] + "_S_D_0") if d == "REF" else ours(name, d or "data/rectified")
            imgs.append(crop(img, b))
            txt.append(reading(folder, run, f"{name}_rectified.png", box))
        rows.append((imgs, txt))
    return rows


def glare_crops():
    runs = [("untreated", "ZONES", "untreated (zones)"), ("classical", "out/classical", "classical"), ("dl_zeroshot", "out/dl_zeroshot", "DocSHRNet, zero-shot"),
            ("dl_ft", "out/dl_ft", "DocSHRNet, fine-tuned"), ("reference", "REF", "reference")]
    rows = region_rows([("F03_S_G_1", 16, (300, 280, 780, 600)), ("F04_S_G_1", 10, (180, 95, 660, 415))], runs, "glare/box_{}")
    # a MIDV-2020 photo; its zones use the tone-matched scan mask
    p = pd.read_csv("out/pairs_midv.csv")
    r = p[p.card_id == "fin_id_68"].iloc[0]
    img, f = load(r.path), Path(r.path).name
    b = (285, 105, 285 + 510, 105 + 340)
    imgs, txt = [], []
    for run, d, _ in runs:
        im = (zones(img, cv2.imread(r.glare_mask_path, 0) > 0) if d == "ZONES" else load(r.ref_path) if d == "REF"
              else load(f"out/midv/{run}/{f}"))
        imgs.append(crop(im, b))
        txt.append(reading("glare/box_{}_midv", run, f, MIDV_BOX))
    panel_grid("glare_crops", rows + [(imgs, txt)], [r[2] for r in runs], FULL)


MIDV_BOX = 6  # date of issue


def chain_crops():
    runs = [("untreated", None, "untreated"), ("dl", "out/dl_ft", "DL deglare"), ("restormer", "out/chain/restormer", "Restormer"),
            ("restormer_dl", "out/chain/restormer_dl", "Restormer → DL"), ("dl_restormer", "out/chain/dl_restormer", "DL → Restormer"),
            ("reference", "REF", "reference")]
    rows = region_rows([("F03_B_G_0", 10, (290, 180, 770, 500)), ("F04_B_G_1", 15, (310, 220, 790, 540))], runs, "joint/box_{}_chain")
    panel_grid("chain_crops", rows, [r[2] for r in runs], FULL)


def fusion(clip="fin_id_95_mid"):
    from glare.fuse_frames import glare_stack
    from glare.prepare_midv_clips import load_stack
    p = pd.read_csv("out/pairs_midv_clips.csv")
    row = p[p.path.str.endswith(f"/{clip}.png")].iloc[0]
    fr = pd.read_csv("out/midv_clips_frames.csv")
    key = load(row.path)
    h, w = key.shape[:2]
    idx, stack, clipped, valid = load_stack(fr[fr.card_id == row.card_id].copy(), Path("data/external/midv2020"), row.key_idx, (w, h))
    G = glare_stack(stack, clipped, valid)
    kp = int(np.flatnonzero(idx == row.key_idx)[0])
    gk = G[kp] & valid[kp]
    own = np.array([(G[k] & valid[k]).mean() for k in range(len(idx))])
    cand = [k for k in range(len(idx)) if k != kp and own[k] >= 0.5 * own[kp]] or [k for k in range(len(idx)) if k != kp]
    ko = max(cand, key=lambda k: (~G[k])[gk].mean())
    other = cv2.cvtColor(stack[ko], cv2.COLOR_BGR2RGB)
    imgs = [key, other, load(f"out/midv_clips/best_field/{clip}.png"), load(f"out/midv_clips/fused/{clip}.png"), load(row.ref_path)]
    panel_grid("fusion", [(imgs, None)], ["key frame", "another aligned frame", "best frame per field", f"fused ({len(idx)} frames)", "scan (reference)"],
               FULL, text_lines=0)


def forest(name, panels, methods, pc, width, xlabel):
    """panels: [(title, {engine: folder}, split, cond, metric)]; one row per method, one dot per engine."""
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(width, 0.62 + 0.27 * len(methods)), sharex=True, sharey=True)
    for ax, (title, part, split, cond, metric) in zip(np.atleast_1d(axs), panels):
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for k, (e, (_, col)) in enumerate(ENGINE.items()):
            for i, (m, _) in enumerate(methods):
                r = pc[(pc.folder == part.format(e)) & (pc.base == "untreated") & (pc.new == m) & (pc.split == split)
                       & (pc.condition == cond) & (pc.metric == metric)]
                if not len(r):
                    continue
                r = r.iloc[0]
                y = i + (-0.15 if k == 0 else 0.15)
                ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(r.change, y, "o", ms=5, color=col, mec="white", mew=0.8, zorder=3)
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_yticks(range(len(methods)), [lab for _, lab in methods])
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.tick_params(length=0, labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    hs = [plt.Line2D([], [], color=c, marker="o", ms=5, lw=1.6, mec="white", mew=0.8, label=lab) for lab, c in ENGINE.values()]
    fig.subplots_adjust(left=0.175 * 7 / width, right=0.985, top=1 - 0.2 / fig.get_figheight(), bottom=0.36 / fig.get_figheight(), wspace=0.12)
    fig.legend(handles=hs, loc="lower left", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0, 0), handlelength=1.6, borderaxespad=0.1)
    fig.text(0.58, 0.02, xlabel, ha="center", va="bottom", fontsize=7)
    save(fig, name)


def paired():
    return pd.read_csv("report/tables/paired_changes.csv")


def halo_changes():
    z = "char_acc_halo"
    forest("halo_changes", [("our test photos (n = 7)", "glare/box_{}", "test", "S_G", z), ("MIDV photos (n = 91)", "glare/box_{}_midv", "midv", "S_G", z),
                            ("MIDV clips, key frames (n = 73)", "glare/box_{}_midv_clips", "keymid", "S_G", z)],
           [("classical", "classical"), ("dl_zeroshot", "DocSHRNet, zero-shot"), ("dl_ft", "DocSHRNet, fine-tuned")], paired(), FULL,
           "change of character accuracy in the halo vs untreated (95% CI)")


def joint_changes():
    z = "char_acc_all"
    forest("joint_changes", [("our blur + glare test photos (n = 9)", "joint/box_{}_chain", "test", "B_G", z), ("MIDV photos (n = 100)", "joint/box_{}_midv_chain", "midv", "S_G", z)],
           [("classical", "classical deglare"), ("dl", "DL deglare"), ("wiener_ad", "Wiener, adaptive"), ("restormer", "Restormer"),
            ("restormer_dl", "Restormer → DL"), ("dl_restormer", "DL → Restormer")], paired(), FULL,
           "change of character accuracy (all boxes) vs untreated (95% CI)")


def tikz(name):
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", f"{name}.tex"], cwd=OUT, check=True, capture_output=True)
    subprocess.run(["latexmk", "-c", "-quiet", f"{name}.tex"], cwd=OUT, capture_output=True)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r300", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                    f"-sOutputFile={name}.png", f"{name}.pdf"], cwd=OUT, check=True)
    print(f"wrote {OUT}/{name}.pdf, .png")


FIGS = dict(dataset=dataset, blur_crops=blur_crops, glare_crops=glare_crops, chain_crops=chain_crops, fusion=fusion, halo_changes=halo_changes,
            joint_changes=joint_changes, pipeline=lambda: tikz("pipeline"), glare_methods=lambda: tikz("glare_methods"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(FIGS))
    for f in ap.parse_args().only.split(","):
        FIGS[f]()


if __name__ == "__main__":
    main()
