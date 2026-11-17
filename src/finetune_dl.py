#!/usr/bin/env python3
"""
Fine-tune a pretrained image-to-image network (e.g. DocShadowNet) to remove glare, then run it.

Training data: usable S_G images of the TRAIN cards in out/pairs.csv (input = glare image,
target = the card's S_D reference).  Choices made from the analysis of your data:
  * random crops, 70% of them centred on glare (glare covers only ~10% of a card)
  * loss = L1 weighted (x glare_weight) inside the glare region + (1 - SSIM)
    the glare region comes from the reference (training only), so no glare detector is needed
    at inference time and glare without a clipped core is handled too
  * checkpoint selection on val S_G with the same do-no-harm rule as the classical method:
    best ssim_glare among checkpoints whose ssim_outside is within --max-harm of untreated
  * iteration 0 is validated too: how the pretrained weights do before any fine-tuning

Usage
  # smoke test of the whole pipeline with a tiny stand-in network (not for results)
  python finetune_dl.py --model tiny --iters 100 --val-every 50

  # real run (see MODEL below for how to point at the DocShadowNet code)
  python finetune_dl.py --model models.xxx:ClassName --model-repo ~/DocShadow-SD7K \
        --pretrained SD7K.pth --name docshadow --iters 3000 --lr 5e-5

  # run the best checkpoint on all glare images, then evaluate as usual
  python finetune_dl.py --infer --model ... --checkpoint checkpoints/docshadow_best.pt --out-dir out/dl
  (--checkpoint may also be the raw SD7K.pth: a zero-shot run, no fine-tuning)
  python evaluate.py --treated-dir out/dl --name dl

MODEL: --model is either `tiny` or `module:Class` (imported from --model-repo, constructor args as
JSON in --model-kwargs).  The network must map an RGB tensor [B,3,H,W] in [0,1] to the cleaned
tensor of the same shape (a tuple/list output uses its first element).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops fall back to CPU

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import glare_region, ssim_scores


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class TinyNet(nn.Module):
    """Small residual conv net: only for testing the pipeline."""

    def __init__(self, ch=24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(ch, 3, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


def build_model(spec: str, repo: str | None, kwargs_json: str | None) -> nn.Module:
    if spec == "tiny":
        return TinyNet()
    mod_name, cls_name = spec.split(":")
    if repo:
        sys.path.insert(0, str(Path(repo).expanduser()))
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(**json.loads(kwargs_json or "{}"))


def load_weights(model: nn.Module, path: str):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # checkpoint with pickled extras; local trusted file
        ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "params", "net", "model_state_dict"):
        if isinstance(ck, dict) and isinstance(ck.get(key), dict):
            ck = ck[key]
            break
    ck = {(k[7:] if k.startswith("module.") else k): v for k, v in ck.items()}
    res = model.load_state_dict(ck, strict=False)
    print(f"loaded {path}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    if res.missing_keys:
        print("  e.g. missing:", res.missing_keys[:3])
    if res.unexpected_keys:
        print("  e.g. unexpected:", res.unexpected_keys[:3])


def unwrap(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_rgb(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).float() / 255.0


class GlareCrops:
    def __init__(self, df: pd.DataFrame, crop: int, p_glare: float, seed: int):
        self.items, self.crop, self.p_glare = [], crop, p_glare
        self.rng = np.random.default_rng(seed)
        for r in df.itertuples():
            x, t = read_rgb(r.path), read_rgb(r.ref_path)
            mask = glare_region(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(t, cv2.COLOR_RGB2GRAY))
            self.items.append((x, t, mask))
        self.crop = min(crop, *(min(x.shape[:2]) for x, _, _ in self.items))

    def batch(self, bs: int):
        xs, ts, ms = [], [], []
        c = self.crop
        for _ in range(bs):
            x, t, m = self.items[self.rng.integers(len(self.items))]
            h, w = m.shape
            if self.rng.random() < self.p_glare and m.any():
                ys, xs_ = np.nonzero(m)
                k = self.rng.integers(len(ys))
                cy, cx = ys[k], xs_[k]
            else:
                cy, cx = self.rng.integers(h), self.rng.integers(w)
            y0 = int(np.clip(cy - c // 2, 0, h - c))
            x0 = int(np.clip(cx - c // 2, 0, w - c))
            sl = (slice(y0, y0 + c), slice(x0, x0 + c))
            xs.append(to_tensor(x[sl]))
            ts.append(to_tensor(t[sl]))
            ms.append(torch.from_numpy(m[sl].astype(np.float32))[None])
        return torch.stack(xs), torch.stack(ts), torch.stack(ms)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def ssim_t(x, y, win=11, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    g = torch.arange(win, dtype=x.dtype, device=x.device) - win // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(x.shape[1], 1, 1, 1)
    f = lambda z: F.conv2d(F.pad(z, (win // 2,) * 4, mode="reflect"), k, groups=x.shape[1])
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.mean()


def loss_fn(pred, target, mask, glare_weight, ssim_weight):
    w = 1 + glare_weight * mask
    l1 = (w * (pred - target).abs()).sum() / (w.sum() * pred.shape[1])
    return l1 + ssim_weight * (1 - ssim_t(pred, target))


# --------------------------------------------------------------------------
# Inference / validation
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, rgb_u8: np.ndarray, device, mult: int) -> np.ndarray:
    model.eval()
    x = to_tensor(rgb_u8)[None].to(device)
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = unwrap(model(x))[..., :h, :w].clamp(0, 1)
    return (y[0].permute(1, 2, 0).float().cpu().numpy() * 255 + 0.5).astype(np.uint8)  # RGB uint8


def validate(model, val, device, mult):
    """mean (whole, glare, outside) SSIM over the val items"""
    sc = []
    for x, t, m in val:
        y = predict(model, x, device, mult)
        sc.append(ssim_scores(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), cv2.cvtColor(y, cv2.COLOR_RGB2GRAY), m))
    return np.nanmean(np.array(sc), axis=0)


def load_pairs(path, splits, conditions):
    p = pd.read_csv(path)
    return p[p.usable & p.split.isin(splits) & p.condition.isin(conditions)].reset_index(drop=True)


# --------------------------------------------------------------------------
def train(a, device):
    tr = load_pairs(a.pairs, ["train"], a.train_conditions.split(","))
    va = load_pairs(a.pairs, ["val"], ["S_G"])
    if tr.empty or va.empty:
        raise SystemExit(f"need usable train ({len(tr)}) and val ({len(va)}) S_G images in {a.pairs}")
    print(f"device {device}; {len(tr)} train images, {len(va)} val images")
    data = GlareCrops(tr, a.crop, a.p_glare, a.seed)
    val = [(x, t, m) for x, t, m in GlareCrops(va, a.crop, 0, a.seed).items]
    print(f"crop {data.crop}px, batch {a.batch}")

    torch.manual_seed(a.seed)
    model = build_model(a.model, a.model_repo, a.model_kwargs)
    if a.pretrained:
        load_weights(model, a.pretrained)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, a.iters) / a.iters)))

    base = np.nanmean(np.array([ssim_scores(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), m)
                                for x, t, m in val]), axis=0)
    fmt = lambda s: f"whole {s[0]:.4f}  glare {s[1]:.4f}  outside {s[2]:.4f}"
    print(f"untreated val : {fmt(base)}")

    ck_dir = Path(a.ckpt_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    best_key = -1e9

    def run_val(it):
        nonlocal best_key
        s = validate(model, val, device, a.pad_multiple)
        ok = s[2] >= base[2] - a.max_harm
        key = s[1] if ok else -1 + s[2]  # any harmless checkpoint beats any harmful one
        tag = ""
        if key > best_key:
            best_key = key
            torch.save({"state_dict": model.state_dict(), "iter": it, "val": s.tolist(), "args": vars(a)},
                       ck_dir / f"{a.name}_best.pt")
            tag = "  <- best" + ("" if ok else " (but harms outside)")
        print(f"val @ {it:5d}: {fmt(s)}{tag}", flush=True)

    run_val(0)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        model.train()
        x, t, m = (z.to(device) for z in data.batch(a.batch))
        loss = loss_fn(unwrap(model(x)), t, m, a.glare_weight, a.ssim_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            el = time.time() - t0
            print(f"it {it:5d}/{a.iters}  loss {loss.item():.4f}  {el:.0f}s elapsed, ~{el / it * (a.iters - it):.0f}s left",
                  flush=True)
        if it % a.val_every == 0 or it == a.iters:
            run_val(it)
    torch.save({"state_dict": model.state_dict(), "iter": a.iters, "args": vars(a)}, ck_dir / f"{a.name}_last.pt")
    print(f"\nBest checkpoint: {ck_dir / (a.name + '_best.pt')}")


def infer(a, device):
    pairs = load_pairs(a.pairs, a.splits.split(","), a.conditions.split(","))
    model = build_model(a.model, a.model_repo, a.model_kwargs)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "args" in ck and "state_dict" in ck:  # our own checkpoint
        model.load_state_dict(ck["state_dict"])
    else:  # raw pretrained weights, e.g. SD7K.pth: zero-shot run without fine-tuning
        load_weights(model, a.checkpoint)
    model.to(device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, r in enumerate(pairs.itertuples(), 1):
        y = predict(model, read_rgb(r.path), device, a.pad_multiple)
        cv2.imwrite(str(out_dir / Path(r.path).name), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        if k % 10 == 0 or k == len(pairs):
            print(f"  {k}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"Wrote {len(pairs)} images to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv")
    ap.add_argument("--model", default="tiny", help="'tiny' or 'module:Class'")
    ap.add_argument("--model-repo", default=None, help="folder added to sys.path to import the model")
    ap.add_argument("--model-kwargs", default=None, help="JSON constructor kwargs")
    ap.add_argument("--pretrained", default=None, help="pretrained weights (.pth) to start from")
    ap.add_argument("--name", default="dl")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--glare-weight", type=float, default=5.0)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--p-glare", type=float, default=0.7, help="fraction of crops centred on glare")
    ap.add_argument("--train-conditions", default="S_G")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--max-harm", type=float, default=0.002)
    ap.add_argument("--pad-multiple", type=int, default=32, help="pad images to a multiple of this for the net")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default="out/dl")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--conditions", default="S_G,B_G")
    a = ap.parse_args()

    device = get_device()
    if a.infer:
        if not a.checkpoint:
            raise SystemExit("--infer needs --checkpoint")
        return infer(a, device)
    train(a, device)


if __name__ == "__main__":
    main()
