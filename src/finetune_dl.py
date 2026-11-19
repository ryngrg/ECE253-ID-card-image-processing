#!/usr/bin/env python3
"""
Learning-based glare removal: a detection-and-removal network fine-tuned on our ID-card pairs.

Model (--model docshrnet, the default): GlareNetMT
  backbone   DocSHRNet (Xu et al., PRCV 2025), a NAFNet-style U-Net pretrained on the DocHighlight
             dataset of real document specular highlights (third_party/DocSHRNet, weights
             third_party/weights/docshrnet.pth).  The JSHDR network of Fu et al. (CVPR 2021) named in
             the proposal is no longer available (its repository and weights were taken down).
  removal    the pretrained output head: y = x + ending(features)
  detection  a new head on the SAME last decoder features -> glare probability m(x)
  output     m * y + (1 - m) * x      the network only changes pixels it detects as glare, so
             text and photo outside the glare are left as they are (do no harm by design)
  The detection head is initialised so that m ~ 1 everywhere: iteration 0 = the pretrained network.
  --no-gate turns the gate off (plain fine-tuning of the removal output), for the ablation.

Training data: usable S_G images of the TRAIN cards in out/pairs.csv (input = glare image,
target = the card's S_D reference).
  * random crops, --p-glare of them centred on glare (glare covers only ~10% of a card), random flips
  * loss = L1 weighted (x glare_weight) inside the glare region + ssim_weight * (1 - SSIM)
           + det_weight * BCE(m, glare region)
    the glare region comes from the reference (training only); no detector is needed at test time
  * checkpoint selection on val S_G with the do-no-harm rule of the classical method: best
    ssim_glare among checkpoints whose ssim_outside is within --max-harm of untreated
  * iteration 0 is validated too: the pretrained weights before any fine-tuning

Usage
  python finetune_dl.py --name docshr_mt --iters 3000                 # fine-tune (MPS if available)
  python finetune_dl.py --infer --checkpoint checkpoints/docshr_mt_best.pt --out-dir out/dl_ft
  python finetune_dl.py --infer --zero-shot --out-dir out/dl_zeroshot  # pretrained, no fine-tuning
  python evaluate.py --engine paddleocr --treated-dir out/dl_ft --name dl_ft
  --infer also saves the predicted glare masks in OUT_DIR/masks/ (detection analysis).
  Other networks: --model module:Class --model-repo DIR --pretrained W.pth (image-level gate head),
  --model tiny for a quick smoke test of the pipeline.
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

ROOT = Path(__file__).resolve().parent.parent
DOCSHR_REPO = ROOT / "third_party" / "DocSHRNet"
DOCSHR_WEIGHTS = ROOT / "third_party" / "weights" / "docshrnet.pth"


# --------------------------------------------------------------------------
# Models
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


def _det_head(cin, ch=32):
    head = nn.Sequential(
        nn.Conv2d(cin, ch, 3, padding=1), nn.GELU(),
        nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
        nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
        nn.Conv2d(ch, 1, 3, padding=1))
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, 4.0)  # sigmoid(4) = 0.98: starts as the plain removal network
    return head


class GlareNetMT(nn.Module):
    """DocSHRNet backbone shared by a removal head (pretrained) and a detection head (new)."""

    def __init__(self, gate=True):
        super().__init__()
        sys.path.insert(0, str(DOCSHR_REPO))
        from model.docshrnet import DocSHRNet
        self.net = DocSHRNet(img_channel=3, width=32)
        self.gate = gate
        self.det = _det_head(32 + 3)

    def features(self, inp):
        n = self.net
        x = n.intro(inp)
        encs = []
        for encoder, down in zip(n.encoders, n.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = n.middle_blks(x)
        for decoder, up, skip in zip(n.decoders, n.ups, encs[::-1]):
            x = decoder(up(x) + skip)
        return x

    def forward(self, inp):
        H, W = inp.shape[-2:]
        xp = self.net.check_image_size(inp)
        f = self.features(xp)
        y = (self.net.ending(f) + xp)[..., :H, :W]
        logit = self.det(torch.cat([f, xp], 1))[..., :H, :W]
        m = torch.sigmoid(logit)
        out = m * y + (1 - m) * inp if self.gate else y
        return out, logit, y


class GatedWrapper(nn.Module):
    """Any image-to-image network + an image-level detection head (for --model module:Class)."""

    def __init__(self, net, gate=True):
        super().__init__()
        self.net, self.gate, self.det = net, gate, _det_head(9)

    def forward(self, inp):
        y = unwrap(self.net(inp))
        logit = self.det(torch.cat([inp, y, y - inp], 1))
        m = torch.sigmoid(logit)
        return (m * y + (1 - m) * inp if self.gate else y), logit, y


def unwrap(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _clean_state(ck):
    for key in ("model_state_dict", "state_dict", "model", "params", "net"):
        if isinstance(ck, dict) and isinstance(ck.get(key), dict):
            ck = ck[key]
            break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in ck.items()}


def load_backbone_weights(module: nn.Module, path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    res = module.load_state_dict(_clean_state(ck), strict=False)
    print(f"loaded {path}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    if res.missing_keys:
        print("  e.g. missing:", res.missing_keys[:3])


def build_model(a) -> nn.Module:
    """Returns a module whose forward gives (output, mask_logit or None, raw_removal)."""
    gate = not a.no_gate
    if a.model == "docshrnet":
        m = GlareNetMT(gate=gate)
        load_backbone_weights(m.net, a.pretrained or DOCSHR_WEIGHTS)
        return m
    if a.model == "tiny":
        net = TinyNet()
    else:
        mod_name, cls_name = a.model.split(":")
        if a.model_repo:
            sys.path.insert(0, str(Path(a.model_repo).expanduser()))
        net = getattr(importlib.import_module(mod_name), cls_name)(**json.loads(a.model_kwargs or "{}"))
        if a.pretrained:
            load_backbone_weights(net, a.pretrained)
    return GatedWrapper(net, gate=gate)


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


def load_items(df: pd.DataFrame):
    items = []
    for r in df.itertuples():
        x, t = read_rgb(r.path), read_rgb(r.ref_path)
        mask = glare_region(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(t, cv2.COLOR_RGB2GRAY))
        items.append((x, t, mask))
    return items


class GlareCrops:
    def __init__(self, items, crop: int, p_glare: float, seed: int, flip: bool = True):
        self.items, self.p_glare, self.flip = items, p_glare, flip
        self.rng = np.random.default_rng(seed)
        self.crop = min(crop, *(min(x.shape[:2]) for x, _, _ in items))

    def batch(self, bs: int):
        xs, ts, ms = [], [], []
        c = self.crop
        for _ in range(bs):
            x, t, m = self.items[self.rng.integers(len(self.items))]
            h, w = m.shape
            if self.rng.random() < self.p_glare and m.any():
                ys, xs_ = np.nonzero(m)
                k = self.rng.integers(len(ys))
                cy, cx = ys[k] + self.rng.integers(-c // 4, c // 4 + 1), xs_[k] + self.rng.integers(-c // 4, c // 4 + 1)
            else:
                cy, cx = self.rng.integers(h), self.rng.integers(w)
            y0 = int(np.clip(cy - c // 2, 0, h - c))
            x0 = int(np.clip(cx - c // 2, 0, w - c))
            sl = (slice(y0, y0 + c), slice(x0, x0 + c))
            xc, tc, mc = x[sl], t[sl], m[sl].astype(np.float32)
            if self.flip and self.rng.random() < 0.5:
                xc, tc, mc = xc[:, ::-1], tc[:, ::-1], mc[:, ::-1]
            if self.flip and self.rng.random() < 0.5:
                xc, tc, mc = xc[::-1], tc[::-1], mc[::-1]
            xs.append(to_tensor(xc))
            ts.append(to_tensor(tc))
            ms.append(torch.from_numpy(np.ascontiguousarray(mc))[None])
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


def loss_fn(out, logit, target, mask, a):
    w = 1 + a.glare_weight * mask
    l1 = (w * (out - target).abs()).sum() / (w.sum() * out.shape[1])
    loss = l1 + a.ssim_weight * (1 - ssim_t(out, target))
    det = F.binary_cross_entropy_with_logits(logit, mask) if logit is not None else torch.zeros((), device=out.device)
    return loss + a.det_weight * det, det


# --------------------------------------------------------------------------
# Inference / validation
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, rgb_u8: np.ndarray, device, mult: int):
    """-> (output RGB uint8, glare probability float32 in [0,1])"""
    model.eval()
    x = to_tensor(rgb_u8)[None].to(device)
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    out, logit, _ = model(x)
    y = out[..., :h, :w].clamp(0, 1)
    m = torch.sigmoid(logit[..., :h, :w])[0, 0].float().cpu().numpy()
    return (y[0].permute(1, 2, 0).float().cpu().numpy() * 255 + 0.5).astype(np.uint8), m


def validate(model, val, device, mult):
    """mean (whole, glare, outside) SSIM and detection IoU over the val items"""
    sc, ious = [], []
    for x, t, m in val:
        y, pm = predict(model, x, device, mult)
        sc.append(ssim_scores(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), cv2.cvtColor(y, cv2.COLOR_RGB2GRAY), m))
        pb = pm > 0.5
        ious.append((pb & m).sum() / max((pb | m).sum(), 1))
    return np.nanmean(np.array(sc), axis=0), float(np.mean(ious))


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
    data = GlareCrops(load_items(tr), a.crop, a.p_glare, a.seed)
    val = load_items(va)
    print(f"crop {data.crop}px, batch {a.batch}, gate {'off' if a.no_gate else 'on'}")

    torch.manual_seed(a.seed)
    model = build_model(a).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("det.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("det.")]
    if a.freeze_encoder and isinstance(model, GlareNetMT):
        for mod in [model.net.intro, model.net.encoders, model.net.downs]:
            for p_ in mod.parameters():
                p_.requires_grad_(False)
        body = [p for p in body if p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, a.iters) / a.iters)))

    base = np.nanmean(np.array([ssim_scores(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), m)
                                for x, t, m in val]), axis=0)
    fmt = lambda s: f"whole {s[0]:.4f}  glare {s[1]:.4f}  outside {s[2]:.4f}"
    print(f"untreated val : {fmt(base)}")

    ck_dir = Path(a.ckpt_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    best_key, log = -1e9, []

    def run_val(it):
        nonlocal best_key
        s, iou = validate(model, val, device, a.pad_multiple)
        ok = s[2] >= base[2] - a.max_harm
        key = s[1] if ok else -1 + s[2]  # any harmless checkpoint beats any harmful one
        tag = ""
        if key > best_key:
            best_key = key
            torch.save({"state_dict": model.state_dict(), "iter": it, "val": s.tolist(), "args": vars(a)},
                       ck_dir / f"{a.name}_best.pt")
            tag = "  <- best" + ("" if ok else " (but harms outside)")
        log.append(dict(iter=it, ssim=s[0], ssim_glare=s[1], ssim_outside=s[2], det_iou=iou))
        pd.DataFrame(log).to_csv(ck_dir / f"{a.name}_val_log.csv", index=False)
        print(f"val @ {it:5d}: {fmt(s)}  det IoU {iou:.3f}{tag}", flush=True)

    run_val(0)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        model.train()
        x, t, m = (z.to(device) for z in data.batch(a.batch))
        out, logit, _ = model(x)
        loss, det = loss_fn(out, logit, t, m, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            el = time.time() - t0
            print(f"it {it:5d}/{a.iters}  loss {loss.item():.4f} (det {det.item():.3f})  "
                  f"{el:.0f}s elapsed, ~{el / it * (a.iters - it):.0f}s left", flush=True)
        if it % a.val_every == 0 or it == a.iters:
            run_val(it)
    torch.save({"state_dict": model.state_dict(), "iter": a.iters, "args": vars(a)}, ck_dir / f"{a.name}_last.pt")
    print(f"\nBest checkpoint: {ck_dir / (a.name + '_best.pt')}")


def infer(a, device):
    pairs = load_pairs(a.pairs, a.splits.split(","), a.conditions.split(","))
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)  # our own checkpoint
        saved = ck.get("args", {})
        a.no_gate = saved.get("no_gate", a.no_gate) if not a.no_gate else True
        a.model = saved.get("model", a.model)
        a.pretrained = None
        model = build_model(a)
        model.load_state_dict(ck["state_dict"])
        print(f"checkpoint {a.checkpoint} (iter {ck.get('iter')}), gate {'off' if a.no_gate else 'on'}")
    elif a.zero_shot:
        a.no_gate = True  # pretrained removal output, untouched
        model = build_model(a)
        print("zero-shot: pretrained weights, no gate, no fine-tuning")
    else:
        raise SystemExit("--infer needs --checkpoint or --zero-shot")
    model.to(device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # cv2.imwrite fails silently on a missing folder
    save_masks = not a.zero_shot and not a.no_gate  # an untrained/unused head gives meaningless masks
    if save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, r in enumerate(pairs.itertuples(), 1):
        y, m = predict(model, read_rgb(r.path), device, a.pad_multiple)
        cv2.imwrite(str(out_dir / Path(r.path).name), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        if save_masks:
            cv2.imwrite(str(out_dir / "masks" / Path(r.path).name), (m * 255 + 0.5).astype(np.uint8))
        if k % 10 == 0 or k == len(pairs):
            print(f"  {k}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"Wrote {len(pairs)} images to {out_dir}/" + (f" (masks in {out_dir}/masks/)" if save_masks else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs.csv")
    ap.add_argument("--model", default="docshrnet", help="'docshrnet', 'tiny' or 'module:Class'")
    ap.add_argument("--model-repo", default=None, help="folder added to sys.path to import the model")
    ap.add_argument("--model-kwargs", default=None, help="JSON constructor kwargs")
    ap.add_argument("--pretrained", default=None, help="pretrained weights (default for docshrnet: "
                                                        "third_party/weights/docshrnet.pth)")
    ap.add_argument("--no-gate", action="store_true", help="output the removal branch directly (ablation)")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--name", default="dl")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--glare-weight", type=float, default=5.0)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--det-weight", type=float, default=0.1)
    ap.add_argument("--p-glare", type=float, default=0.7, help="fraction of crops centred on glare")
    ap.add_argument("--train-conditions", default="S_G")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--max-harm", type=float, default=0.002)
    ap.add_argument("--pad-multiple", type=int, default=16, help="pad images to a multiple of this for the net")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--zero-shot", action="store_true", help="--infer with the pretrained weights only")
    ap.add_argument("--out-dir", default="out/dl")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--conditions", default="S_G,B_G")
    a = ap.parse_args()

    device = get_device()
    if a.infer:
        return infer(a, device)
    train(a, device)


if __name__ == "__main__":
    main()
