#!/usr/bin/env python3
"""
Where do the glare methods act?  IoU of each method's glare region with the reference-derived one.

  iou_dl               DL detection head (finetune_dl.py --infer saves it in OUT/masks/, > 0.5) vs evaluate.glare_region
  iou_classical        classical core + veil region (seeds | a > a_min, the region the method may change) vs glare_region
  iou_dl_vs_classical  the two method regions against each other

The report cites the DL value on test S_G (0.30) against train S_G (0.59): the detection head overfits.

Usage:  python src/glare/detection_iou.py [--dl-masks out/dl_ft/masks] [--out results/glare/detection_iou.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from common.evaluate import glare_region
from glare.classical_glare import DEFAULTS, load_pairs, remove_glare


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv", type=Path)
    ap.add_argument("--params", default="params/glare/classical_params.json", type=Path)
    ap.add_argument("--dl-masks", default="out/dl_ft/masks", type=Path)
    ap.add_argument("--out", default="results/glare/detection_iou.csv", type=Path)
    a = ap.parse_args()
    p = {**DEFAULTS, **json.loads(a.params.read_text())}

    rows = []
    for r in load_pairs(a.pairs).itertuples():
        name = Path(r.path).name
        img, ref = cv2.imread(r.path), cv2.imread(r.ref_path)
        glare = glare_region(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        _, core, veil = remove_glare(img, p, return_maps=True)
        classical = core | (veil > p["a_min"])
        dl = cv2.imread(str(a.dl_masks / name), cv2.IMREAD_GRAYSCALE) > 127
        rows.append(dict(image=name, split=r.split, condition=r.condition, iou_classical=iou(classical, glare),
                         iou_dl=iou(dl, glare), iou_dl_vs_classical=iou(dl, classical)))
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(d.groupby(["split", "condition"])[["iou_classical", "iou_dl", "iou_dl_vs_classical"]].mean().round(3).to_string())
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
