#!/usr/bin/env python3
"""
Tables of the report and slides, built from results/ (never typed by hand).

Writes report/tables/{name}.tex (booktabs tabular, no caption), report/tables/tables.md (all of them, for reading and
slides) and report/tables/paired_changes.csv (every paired change shown anywhere, with CI, n and better / worse counts).

Accuracy = character accuracy pooled over the boxes of a zone (box_ocr.py summary.csv).
Change   = mean of the per-image differences, with a 95% bootstrap CI over images (compare_runs.boot_ci), so it differs a
           little from the difference of two pooled accuracies.  Bold = the CI excludes 0.

Usage:  python src/report/make_tables.py
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common.compare_runs import boot_ci

RES, OUT = Path("results"), Path("report/tables")
ENGINES = [("paddle", "PaddleOCR"), ("tesseract", "Tesseract")]
KEYS = ["image", "card_id", "condition", "split"]
LOG, MD = [], []


@lru_cache(None)
def read(f):
    return pd.read_csv(f)


def acc(folder, method, split, cond, zone):
    s = read(RES / folder / "summary.csv")
    r = s[(s.method == method) & (s.split == split) & (s.condition == cond) & (s.zone == zone)]
    assert len(r) == 1, (folder, method, split, cond, zone, len(r))
    return r.iloc[0].char_acc, int(r.iloc[0].n_boxes)


def change(folder, new, split, cond, metric, base="untreated"):
    """dict(change, lo, hi, n, better, worse) of one metric column ('char_acc_halo', 'ssim', ...)."""
    b, n = (read(RES / folder / f"{m}_per_image.csv") for m in (base, new))
    m = b.merge(n, on=KEYS, suffixes=("_b", "_n"))
    m = m[(m.split == split) & (m.condition == cond)]
    d = (m[f"{metric}_n"] - m[f"{metric}_b"]).dropna().to_numpy()
    lo, hi = boot_ci(d)
    r = dict(folder=folder, split=split, condition=cond, metric=metric, base=base, new=new, n=len(d), change=d.mean(),
             ci_low=lo, ci_high=hi, better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()))
    LOG.append(r)
    return r


def f3(x):
    return f"{x:.3f}".replace("0.", ".", 1) if 0 <= x < 1 else f"{x:.3f}"


def signed(x):
    s = f"{abs(x):.3f}".replace("0.", ".", 1) if abs(x) < 1 else f"{abs(x):.3f}"
    return ("$-$" if x < 0 and s != ".000" else "+") + s


def ci(r, with_ci=True):
    t = signed(r["change"])
    if with_ci:
        t += f" [{signed(r['ci_low'])}, {signed(r['ci_high'])}]".replace("+", "")
    return rf"\textbf{{{t}}}" if r["ci_low"] > 0 or r["ci_high"] < 0 else t


def best(values, skip=()):
    """Column-wise bold of the best value (rows in skip, e.g. the reference, do not compete)."""
    top = max(v for i, v in enumerate(values) if i not in skip and v is not None)
    return ["--" if v is None else (rf"\textbf{{{f3(v)}}}" if i not in skip and f3(v) == f3(top) else f3(v)) for i, v in enumerate(values)]


def write(name, title, header, rows, spec, groups=None, note=""):
    """groups: [(label, n columns)] for a spanning first header line; a row that is None is a midrule."""
    t = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    if groups:
        t.append(" & ".join(rf"\multicolumn{{{n}}}{{c}}{{{g}}}" if g else "" for g, n in groups) + r" \\")
        c, rules = 1, []
        for g, n in groups:
            if g:
                rules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
            c += n
        t.append("".join(rules))
    t += [" & ".join(header) + r" \\", r"\midrule"]
    t += [r"\midrule" if r is None else " & ".join(r) + r" \\" for r in rows]
    t += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(t))

    def md(s):
        s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s)
        return s.replace("$-$", "−").replace(r"$\rightarrow$", "→").replace("--", "–").replace(r"\,", " ").replace("$", "").replace(r"\%", "%").replace(r"\_", "_").replace(r"\quad", "–")
    h = header
    if groups:
        lab = [g for g, n in groups for _ in range(n)]
        h = [f"{g} {x}".strip() if g else x for g, x in zip(lab, header)]
    MD.append(f"## {name}: {title}\n" + (note + "\n" if note else "") + "\n| " + " | ".join(map(md, h)) + " |\n|" + "---|" * len(h) + "\n"
              + "\n".join("| " + " | ".join(map(md, r)) + " |" for r in rows if r) + "\n")


# ------------------------------------------------------------------ data
def t_dataset():
    p, b, s = read(Path("out/pairs.csv")), read(Path("out/pairs_blur.csv")), read(Path("split.csv"))
    p, b = p[p.usable], b[b.usable]
    rows = []
    for sp in ("train", "val", "test"):
        n = [int((s.split == sp).sum())] * 2 + [int((b.split == sp).sum())]
        n += [int(((p.split == sp) & (p.condition == c)).sum()) for c in ("S_G", "B_G")]
        rows.append([sp] + n)
    rows.append(["total"] + [sum(r[i] for r in rows) for i in range(1, 6)])
    rows = [[str(x) for x in r] for r in rows]
    rows.insert(3, None)
    write("dataset", "our photos by split (usable images)", ["", "cards", "reference", "blur", "glare", "blur + glare"], rows, "lrrrrr",
          note="reference = sharp + diffuse (S_D, 1 per card), blur = B_D, glare = S_G, blur + glare = B_G. One of the 95 glare photos is unusable.")


# ------------------------------------------------------------------ blur and joint
def t_levels_and_changes(name, title, part, ssim, split, cond, runs, note=""):
    """Rows = methods; per engine: accuracy over all boxes and paired change; optional SSIM change."""
    ref = len(runs) - 1
    cols = []
    for e, _ in ENGINES:
        cols.append(best([acc(part.format(e), m, split, cond, "all")[0] for m, _ in runs], skip=(ref,)))
        cols.append(["--" if m in ("untreated", "reference") else ci(change(part.format(e), m, split, cond, "char_acc_all")) for m, _ in runs])
    if ssim:
        cols.append(["--" if m in ("untreated", "reference") else ci(change(ssim, m, split, cond, "ssim")) for m, _ in runs])
    rows = [[lab] + [c[i] for c in cols] for i, (_, lab) in enumerate(runs)]
    rows.insert(ref, None)
    groups = [("", 1)] + [(lab, 2) for _, lab in ENGINES] + ([("SSIM", 1)] if ssim else [])
    write(name, title, ["method"] + ["acc.", "change [95\\% CI]"] * 2 + (["change [95\\% CI]"] if ssim else []), rows,
          "l" + "rl" * 2 + ("l" if ssim else ""), groups, note)


A = r" $\rightarrow$ "


def t_blur():
    n = acc("blur/box_paddle_blur", "untreated", "test", "B_D", "all")[1]
    t_levels_and_changes("blur", "deblurring, test B_D", "blur/box_{}_blur", "blur/ssim_blur", "test", "B_D",
                         [("untreated", "untreated"), ("wiener", "Wiener, fixed"), ("wiener_ad", "Wiener, adaptive"),
                          ("restormer_pre", "Restormer, pretrained"), ("restormer", "Restormer, fine-tuned"), ("reference", "reference")],
                         f"n = 4 images, {n} boxes. Changes vs untreated.")


def t_joint():
    runs = [("untreated", "untreated"), ("classical", "classical deglare"), ("dl", "DL deglare"), ("wiener", "Wiener, fixed"), ("wiener_ad", "Wiener, adaptive"),
            ("restormer_pre", "Restormer, pretrained"), ("restormer", "Restormer (ft)"),
            ("wiener_ad_classical", "Wiener, ad." + A + "classical"), ("classical_wiener_ad", "classical" + A + "Wiener, ad."),
            ("restormer_classical", "Restormer" + A + "classical"), ("restormer_dl", "Restormer" + A + "DL"),
            ("dl_restormer", "DL" + A + "Restormer"), ("reference", "reference")]
    n = acc("joint/box_paddle_chain", "untreated", "test", "B_G", "all")[1]
    t_levels_and_changes("joint_bg", "joint, test B_G", "joint/box_{}_chain", "joint/ssim_chain", "test", "B_G", runs,
                         f"n = 9 images, {n} boxes. Changes vs untreated. A → B = A first. DL = fine-tuned DocSHRNet.")
    t_levels_and_changes("joint_sg", "do no harm on sharp images, test S_G", "joint/box_{}_chain", "joint/ssim_chain", "test", "S_G",
                         [r for r in runs if r[0] in ("untreated", "dl", "wiener_ad", "restormer", "restormer_dl", "dl_restormer", "reference")],
                         "n = 8 images.")
    n = acc("joint/box_paddle_midv_chain", "untreated", "midv", "S_G", "all")[1]
    t_levels_and_changes("joint_midv", "MIDV-2020 photos", "joint/box_{}_midv_chain", None, "midv", "S_G",
                         [r for r in runs if r[0] not in ("restormer_pre", "restormer_classical")], f"n = 100 images, {n} boxes, external.")
    # paper version: paired changes only, both data sets in one table
    short = [r for r in runs if r[0] not in ("restormer_pre", "restormer_classical", "wiener_ad_classical", "classical_wiener_ad")]
    sets = [("joint/box_{}_chain", "test", "B_G"), ("joint/box_{}_midv_chain", "midv", "S_G")]
    rows = []
    for m, lab in short:
        if m in ("untreated", "reference"):
            rows.append([lab + " (accuracy)"] + [f3(acc(p.format(e), m, s, c, "all")[0]) for p, s, c in sets for e, _ in ENGINES])
        else:
            rows.append([lab] + [ci(change(p.format(e), m, s, c, "char_acc_all")) for p, s, c in sets for e, _ in ENGINES])
    rows.insert(len(rows) - 1, None)
    rows.insert(1, None)
    write("joint", "joint, paired changes (paper version)", ["method"] + [lab for _, lab in ENGINES] * 2, rows, "lllll",
          [("", 1), ("blur + glare, our test ($n=9$)", 2), ("glare, MIDV photos ($n=100$)", 2)])
    # the two questions about the chain, answered directly
    rows = []
    for part, split, cond, lab in (("joint/box_{}_chain", "test", "B_G", "B\\_G test"), ("joint/box_{}_midv_chain", "midv", "S_G", "MIDV photos")):
        for new, base, q in (("restormer_dl", "restormer", "DL after Restormer"), ("restormer_dl", "dl", "Restormer before DL"),
                             ("restormer_dl", "dl_restormer", "deblur first vs deglare first")):
            rows.append([lab, q] + [ci(change(part.format(e), new, split, cond, "char_acc_all", base)) for e, _ in ENGINES])
    write("joint_steps", "what each step of the chain adds (character accuracy, all boxes)", ["data", "comparison", "PaddleOCR", "Tesseract"], rows, "llll")


# ------------------------------------------------------------------ glare
GLARE = [("untreated", "untreated"), ("classical", "classical"), ("dl_zeroshot", "DL, zero-shot"), ("dl_ft", "DL, fine-tuned"), ("reference", "reference")]
SETS = [("our test", "glare/box_{}", "test"), ("MIDV photos", "glare/box_{}_midv", "midv")]


def t_glare():
    zones = ("core", "halo", "outside")
    cols, head, groups, note = [], [], [("", 1)], []
    for e, elab in ENGINES:
        for lab, part, split in SETS:
            groups.append((f"{elab}, {lab}", 3))
            for z in zones:
                cols.append(best([acc(part.format(e), m, split, "S_G", z)[0] for m, _ in GLARE], skip=(4,)))
                head.append(z)
    for lab, part, split in SETS:
        note.append(f"{lab}: " + " / ".join(f"{acc(part.format('paddle'), 'untreated', split, 'S_G', z)[1]} {z}" for z in zones) + " boxes")
    rows = [[lab] + [c[i] for c in cols] for i, (_, lab) in enumerate(GLARE)]
    rows.insert(4, None)
    write("glare_zones", "glare, character accuracy per zone", ["method"] + head, rows, "l" + "r" * 12, groups, "; ".join(note) + ".")

    # one-column paper version: core and halo only
    cols = [best([acc(part.format(e), m, split, "S_G", z)[0] for m, _ in GLARE], skip=(4,))
            for e, _ in ENGINES for _, part, split in SETS for z in ("core", "halo")]
    rows = [[lab] + [c[i] for c in cols] for i, (_, lab) in enumerate(GLARE)]
    rows.insert(4, None)
    write("glare_small", "glare, core and halo (paper version)", ["method"] + ["core", "halo"] * 4, rows, "l" + "r" * 8,
          [("", 1)] + [(f"{e}, {s}", 2) for e in ("P", "T") for s in ("ours", "MIDV")])

    rows = []
    for m, mlab in GLARE[1:4]:
        r = [mlab]
        for e, _ in ENGINES:
            for _, part, split in SETS + [("", "glare/box_{}_midv_clips", "keymid")]:
                r.append(ci(change(part.format(e), m, split, "S_G", "char_acc_halo")))
        rows.append(r)
    ns = [change(part.format("paddle"), "dl_ft", split, "S_G", "char_acc_halo")["n"] for _, part, split in SETS + [("", "glare/box_{}_midv_clips", "keymid")]]
    write("glare_halo_changes", "glare, paired change of halo accuracy vs untreated", ["method"] + ["our test", "MIDV photos", "MIDV clips"] * 2, rows,
          "l" + "l" * 6, [("", 1), ("PaddleOCR", 3), ("Tesseract", 3)], f"n (images with a halo box) = {ns[0]} / {ns[1]} / {ns[2]}; clips: keymid frames, single-image methods.")


def t_ablation():
    """With and without our modifications; SSIM on the whole card and PaddleOCR / fixed boxes in the glare."""
    runs = [("untreated", "untreated"), ("classical_decay", "classical, distance-decay veil"), ("classical", "classical, statistics veil (ours)"),
            ("classical_veil", "\\quad veil only"), ("classical_inpaint", "\\quad core fill only"), ("classical_telea", "\\quad Telea core fill"),
            ("dl_zeroshot", "DocSHRNet, zero-shot"), ("dl_ft", "DocSHRNet, fine-tuned (ours)")]
    s = read(RES / "glare/whole_card/all_methods.csv")
    s = s[(s.split == "test") & (s.condition == "S_G")].set_index("method")
    cols = [best([s.loc[m, c] for m, _ in runs]) for c in ("ssim", "ssim_glare", "ssim_outside", "f1_gt")]
    cols += [best([acc("glare/box_paddle", m, "test", "S_G", z)[0] for m, _ in runs]) for z in ("core", "halo")]
    rows = [[lab] + [c[i] for c in cols] for i, (_, lab) in enumerate(runs)]
    write("ablation", "glare with / without our modifications, test S_G (n = 8)",
          ["method", "card", "glare", "outside", "word F1", "core", "halo"], rows, "lrrrrrr",
          [("", 1), ("SSIM", 3), ("card OCR", 1), ("box acc.", 2)], "OCR columns: PaddleOCR (whole-card word F1 vs ground truth; fixed-box character accuracy).")


def t_modifications():
    """Each modification against the version without it: the course's 'with and without' question."""
    S = "char_acc_{}".format
    spec = [  # label, folder, base, new, split, cond, zone, SSIM folder, (SSIM base, new), SSIM metric
        ("Wiener: fixed" + A + "adaptive", "B", "blur/box_{}_blur", "wiener", "wiener_ad", "test", "B_D", "all", "blur/ssim_blur", "ssim"),
        ("Restormer: pretrained" + A + "fine-tuned", "B", "blur/box_{}_blur", "restormer_pre", "restormer", "test", "B_D", "all", "blur/ssim_blur", "ssim"),
        ("classical: decay" + A + "statistics veil", "G, halo", "glare/box_{}", "classical_decay", "classical", "test", "S_G", "halo", "glare/whole_card", "ssim_glare"),
        ("classical: Telea" + A + "flat core fill", "G, core", "glare/box_{}", "classical_telea", "classical", "test", "S_G", "core", "glare/whole_card", "ssim_glare"),
        ("DocSHRNet: zero-shot" + A + "fine-tuned", "G, halo", "glare/box_{}", "dl_zeroshot", "dl_ft", "test", "S_G", "halo", "glare/whole_card", "ssim_glare"),
        ("DocSHRNet: zero-shot" + A + "fine-tuned", "MIDV, halo", "glare/box_{}_midv", "dl_zeroshot", "dl_ft", "midv", "S_G", "halo", None, None)]
    rows = []
    for lab, data, part, base, new, split, cond, zone, sdir, smet in spec:
        rs = [change(part.format(e), new, split, cond, S(zone), base) for e, _ in ENGINES]
        rows.append([lab, f"{data} ({rs[0]['n']})"] + [ci(r) for r in rs] + [ci(change(sdir, new, split, cond, smet, base)) if sdir else "--"])
    write("modifications", "effect of each modification (new minus old version)", ["modification", "test data ($n$)", "PaddleOCR", "Tesseract", "SSIM"], rows, "lllll",
          note="Character accuracy in the named zone (B_D: all boxes); SSIM: whole card for blur, inside the glare for glare. n = images with a box in the zone.")


# ------------------------------------------------------------------ fusion
def t_fusion():
    runs = [("untreated", "key frame, untreated"), ("classical", "classical"), ("dl_ft", "DL, fine-tuned"), ("best_frame", "best single frame"),
            ("best_field", "best frame per field"), ("median", "temporal median"), ("fused", "fused"), ("reference", "reference (scan)")]
    cols, groups, head = [], [("", 1)], []
    for split in ("keymid", "keymax"):
        for e, elab in ENGINES:
            groups.append((f"{split}, {elab}", 3))
            for z in ("core", "halo", "all"):
                cols.append(best([acc(f"glare/box_{e}_midv_clips", m, split, "S_G", z)[0] for m, _ in runs], skip=(7,)))
                head.append(z)
    rows = [[lab] + [c[i] for c in cols] for i, (_, lab) in enumerate(runs)]
    rows.insert(7, None)
    nb = [acc("glare/box_paddle_midv_clips", "untreated", s, "S_G", z)[1] for s in ("keymid", "keymax") for z in ("core", "halo", "all")]
    write("fusion", "multi-frame fusion, 100 MIDV-2020 clips", ["method"] + head, rows, "l" + "r" * 12, groups,
          f"boxes core / halo / all: keymid {nb[0]} / {nb[1]} / {nb[2]}, keymax {nb[3]} / {nb[4]} / {nb[5]}.")
    # one-column paper version
    short = [("untreated", "untreated"), ("classical", "classical"), ("dl_ft", "DL, fine-tuned"), ("best_frame", "best frame"), ("best_field", "best per field"),
             ("median", "median"), ("fused", "fused"), ("reference", "scan")]
    cols = [best([acc(f"glare/box_{e}_midv_clips", m, s, "S_G", z)[0] for m, _ in short], skip=(7,))
            for e, _ in ENGINES for s, z in (("keymax", "core"), ("keymax", "all"), ("keymid", "all"))]
    rows = [[lab] + [c[i] for c in cols] for i, (_, lab) in enumerate(short)]
    rows.insert(7, None)
    write("fusion_small", "multi-frame fusion (paper version)", ["method"] + ["core", "all", "all"] * 2, rows, "l" + "r" * 6,
          [("", 1), ("PaddleOCR", 3), ("Tesseract", 3)], "columns per engine: keymax core, keymax all, keymid all.")
    rows = []
    for base, lab in (("untreated", "fused vs untreated"), ("best_frame", "fused vs best single frame"), ("best_field", "fused vs best frame per field")):
        rows.append([lab] + [ci(change(f"glare/box_{e}_midv_clips", "fused", s, "S_G", f"char_acc_{z}", base))
                             for e, _ in ENGINES for s, z in (("keymax", "core"), ("keymax", "all"), ("keymid", "all"))])
    write("fusion_changes", "multi-frame fusion, paired changes", ["comparison"] + ["keymax core", "keymax all", "keymid all"] * 2, rows, "l" + "l" * 6,
          [("", 1), ("PaddleOCR", 3), ("Tesseract", 3)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for t in (t_dataset, t_blur, t_glare, t_ablation, t_modifications, t_fusion, t_joint):
        t()
    (OUT / "tables.md").write_text("# Tables (generated by src/report/make_tables.py; do not edit)\n\nAccuracy = character accuracy pooled over fixed boxes. "
                                   "Change = mean paired per-image difference [95% bootstrap CI]; bold = CI excludes 0 (changes) or best in column (levels).\n\n" + "\n".join(MD))
    pd.DataFrame(LOG).drop_duplicates().to_csv(OUT / "paired_changes.csv", index=False)
    print(f"{len(MD)} tables -> {OUT}/")


if __name__ == "__main__":
    main()
