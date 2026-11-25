#!/usr/bin/env python3
"""
Fine-tune the official single-image defocus Restormer on our B_D -> S_D pairs (out/pairs_blur.csv).

Port of the blur repo's restormer/scripts/train_restormer.py (--mode real; Mlazazarus/ECE253 @ 2893d66),
with the same recipe and checkpoint format, so deblur.py --weights loads the result:
  official pretrained weights, 128 px patches (75% near a text box), batch 1, random flips / 90-degree
  rotations, L1 on the unclipped RGB prediction, AdamW lr 1e-5, no weight decay, gradient norm clip 1.
Differences from the blur repo
  * data: our rectified, reference-aligned images and split.csv (the pilot used native-resolution crops, other
    splits and MIDV clips; see docs/blur_history/README.md).  Cards are sampled equally, then images within a card;
  * the target is tone-matched to the blurred input (per-channel gain + offset, fitted on low-passed images):
    exposure and white balance differ by up to ~15% between the B_D and S_D photos, and a deblurring
    network must not learn to change them.  --no-match-tone reproduces the original loss;
  * validation = fixed patches on up to 8 text boxes per val image instead of one patch per image.

Only train cards are trained on and only val cards are validated on; test rows are dropped on loading.
Every validation step is saved (step_XXXXXX.pt).  Patch L1 is only a monitor: choose the checkpoint with
  python src/tune_deblur.py restormer --ckpt-dir checkpoints/deblur/restormer_ours

Usage
  python src/finetune_deblur.py                              # 2000 steps -> checkpoints/deblur/restormer_ours/
  python src/finetune_deblur.py --name smoke --steps 4 --validate-every 2
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from deblur_blocks import digest
from deblur_blocks.restormer import (DEFAULT_WEIGHTS, TASK, load_model, read_rgb, save_json, select_device,
                                     to_tensor, unit_rgb)


def load_pairs(pairs_file, split):
    """Usable B_D rows of one split, with the split checked against split.csv."""
    if split not in ("train", "val"):
        raise ValueError("Only train and val may be used for training / tuning.")
    pairs = pd.read_csv(pairs_file)
    pairs = pairs[pairs.usable & (pairs.split == split)].reset_index(drop=True)
    official = pd.read_csv("split.csv").set_index("card_id").split
    if not (pairs.card_id.map(official) == split).all():
        raise ValueError(f"{pairs_file}: split column disagrees with split.csv; rerun prepare_data.py.")
    if pairs.empty:
        raise ValueError(f"{pairs_file}: no usable {split} rows.")
    return pairs


def match_tone(target, source, sigma=8.0):
    """Per-channel gain + offset that maps the (low-passed) target onto the (low-passed) source."""
    t, s = cv2.GaussianBlur(target, (0, 0), sigma), cv2.GaussianBlur(source, (0, 0), sigma)
    out = np.empty_like(target)
    for c in range(3):
        gain, offset = np.polyfit(t[..., c].ravel(), s[..., c].ravel(), 1)
        out[..., c] = target[..., c] * gain + offset
    return np.clip(out, 0, 1)


def load_samples(pairs, layout, tone):
    samples = []
    for r in pairs.itertuples():
        x, y = unit_rgb(read_rgb(r.path)), unit_rgb(read_rgb(r.ref_path))
        if x.shape != y.shape:
            raise ValueError(f"{r.path}: size differs from its reference.")
        boxes = layout[(layout.card_id == r.card_id) & layout.labelled][["x0", "y0", "x1", "y1"]].to_numpy(float)
        samples.append(dict(card_id=r.card_id, path=r.path, ref_path=r.ref_path, input=x,
                            target=match_tone(y, x) if tone else y,
                            centers=np.c_[boxes[:, [0, 2]].mean(1), boxes[:, [1, 3]].mean(1)]))
    return samples


def patch_at(sample, cx, cy, size):
    h, w = sample["input"].shape[:2]
    x = int(np.clip(round(cx - size / 2), 0, w - size))
    y = int(np.clip(round(cy - size / 2), 0, h - size))
    return (sample["input"][y:y + size, x:x + size], sample["target"][y:y + size, x:x + size])


def training_patch(samples, size, rng, p_text):
    cards = sorted({s["card_id"] for s in samples})
    card = cards[int(rng.integers(len(cards)))]
    choices = [s for s in samples if s["card_id"] == card]
    s = choices[int(rng.integers(len(choices)))]
    h, w = s["input"].shape[:2]
    if len(s["centers"]) and rng.random() < p_text:
        cx, cy = s["centers"][int(rng.integers(len(s["centers"])))] + rng.uniform(-size / 4, size / 4, 2)
    else:
        cx, cy = rng.uniform(size / 2, w - size / 2), rng.uniform(size / 2, h - size / 2)
    x, y = patch_at(s, cx, cy, size)
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    x, y = np.rot90(x, k), np.rot90(y, k)
    if flip:
        x, y = x[:, ::-1], y[:, ::-1]
    return x.copy(), y.copy()


def validation_patches(samples, size, per_image=8):
    out = []
    for s in samples:
        h, w = s["input"].shape[:2]
        c = s["centers"] if len(s["centers"]) else np.array([[w / 2, h / 2]])
        for cx, cy in c[np.linspace(0, len(c) - 1, min(per_image, len(c))).round().astype(int)]:
            out.append((s["card_id"], *patch_at(s, cx, cy, size)))
    return out


@torch.inference_mode()
def validate(model, patches, device):
    """Mean over cards of the mean patch L1."""
    model.eval()
    per_card = {}
    for card, x, y in patches:
        loss = float(F.l1_loss(model(to_tensor(x, device)), to_tensor(y, device)).cpu())
        per_card.setdefault(card, []).append(loss)
    return float(np.mean([np.mean(v) for v in per_card.values()]))


def save_checkpoint(path, model, metadata):
    tmp = path.with_suffix(".tmp")
    torch.save({"params": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "restormer_experiment": metadata}, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="out/pairs_blur.csv", type=Path)
    ap.add_argument("--layout", default="results/box_layout.csv", type=Path, help="text boxes for text-centred patches")
    ap.add_argument("--name", default="restormer_ours")
    ap.add_argument("--ckpt-dir", default="checkpoints/deblur", type=Path)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--validate-every", type=int, default=500)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--p-text", type=float, default=0.75, help="fraction of patches centred near a text box")
    ap.add_argument("--no-match-tone", action="store_true", help="raw S_D target (the blur repo's loss)")
    ap.add_argument("--seed", type=int, default=253)
    a = ap.parse_args()
    out = a.ckpt_dir / a.name
    if out.exists():
        ap.error(f"{out} exists: choose a new --name.")
    if a.steps < 1 or a.validate_every < 1 or a.patch_size < 32 or a.patch_size % 8 or not 0 <= a.p_text <= 1:
        ap.error("Need positive steps, a patch size >= 32 divisible by 8, and 0 <= p-text <= 1.")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    layout = pd.read_csv(a.layout, keep_default_na=False)
    layout["labelled"] = layout.labelled.astype(str) == "True"
    tone = not a.no_match_tone
    train = load_samples(load_pairs(a.pairs, "train"), layout, tone)
    val = load_samples(load_pairs(a.pairs, "val"), layout, tone)
    patches = validation_patches(val, a.patch_size)
    device = select_device(a.device)
    model, info = load_model(a.weights, device)
    if info["fine_tuning"] is not None:
        ap.error("Start from the official pretrained weights.")

    out.mkdir(parents=True)
    # the first four keys are required by deblur_blocks.restormer.load_model
    metadata = dict(task=TASK, architecture=info["architecture"], architecture_sha256=info["architecture_sha256"],
                    pretrained_sha256=info["pretrained_sha256"], script="src/finetune_deblur.py",
                    script_sha256=digest(__file__), split_sha256=digest("split.csv"), pairs_sha256=digest(a.pairs),
                    train_images=[s["path"] for s in train], val_images=[s["path"] for s in val],
                    train_cards=sorted({s["card_id"] for s in train}), lr=a.lr, seed=a.seed, steps=a.steps,
                    patch_size=a.patch_size, p_text=a.p_text, match_tone=tone, torch_version=str(torch.__version__),
                    device=str(device))
    base = validate(model, patches, device)
    print(f"{len(train)} train images ({len(metadata['train_cards'])} cards), {len(val)} val images, "
          f"{len(patches)} val patches, device {device}\npretrained: val L1 {base:.5f}", flush=True)
    history = [dict(step=0, val_l1=base)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    losses, start = [], time.perf_counter()
    for step in range(1, a.steps + 1):
        model.train()
        x, y = training_patch(train, a.patch_size, rng, a.p_text)
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(to_tensor(x, device)), to_tensor(y, device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % a.validate_every == 0 or step == a.steps:
            v = validate(model, patches, device)
            history.append(dict(step=step, train_l1=float(np.mean(losses)), val_l1=v,
                                seconds=time.perf_counter() - start))
            losses = []
            save_checkpoint(out / f"step_{step:06d}.pt", model, {**metadata, "step": step, "val_l1": v})
            save_json(out / "run.json", dict(settings=metadata, history=history))
            print(f"step {step}/{a.steps}: train L1 {history[-1]['train_l1']:.5f}, val L1 {v:.5f} "
                  f"({history[-1]['seconds']:.0f}s)", flush=True)
    print(f"Saved {out}/.  Next: python src/tune_deblur.py restormer --ckpt-dir {out}")


if __name__ == "__main__":
    main()
