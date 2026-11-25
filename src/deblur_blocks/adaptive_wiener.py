"""Adaptive Wiener: blur radius and noise level estimated per image (the proposal's Wiener method).

Port of the blur repo's wiener/weiner_filter.py (Mlazazarus/ECE253 @ 2893d66; it was not part of the exported
blocks).  No reference, no ground truth:
  * noise: sigma = MAD of the finest diagonal Haar coefficients; K = noise_scale * sigma^2 / signal variance;
  * radius: every radius 0..max_radius is tried and the one with the best
        Tesseract confidence + sharpness_weight * sharpness - ringing_weight * ringing   is kept.
Radii are in input pixels.  The Haar transform is written out in NumPy, so PyWavelets is not needed.
"""
import numpy as np
from scipy import ndimage

from .wiener import luminance, restore

DEFAULTS = {"max_radius": 8, "noise_scale": 10, "sharpness_weight": 0, "ringing_weight": 4}


def estimate_noise(y):
    """(noise sigma, base Wiener K) of a luminance image in [0, 1]."""
    y = np.pad(y, ((0, y.shape[0] % 2), (0, y.shape[1] % 2)), mode="edge")
    detail = (y[::2, ::2] - y[::2, 1::2] - y[1::2, ::2] + y[1::2, 1::2]) / 2
    sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.67448975)
    signal = max(float(np.var(y)) - sigma ** 2, 1e-8)
    return sigma, float(np.clip(sigma ** 2 / signal, 1e-5, 1))


def gradient_energy(y):
    gx = ndimage.sobel(y, axis=1, mode="reflect") / 8
    gy = ndimage.sobel(y, axis=0, mode="reflect") / 8
    return float(np.mean(gx * gx + gy * gy))


def to_uint8(image):
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def ocr_confidence(rgb_uint8):
    """Mean Tesseract word confidence in [0, 1] (--psm 6, whole card)."""
    import pytesseract
    d = pytesseract.image_to_data(rgb_uint8, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT,
                                  timeout=60)
    conf = [float(c) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    return float(np.mean(conf)) / 100 if conf else 0.0


def candidates(image, noise_scale, max_radius):
    """Yield (uint8 RGB, features) for each radius; image = float RGB in [0, 1]."""
    y = luminance(image)
    k = min(noise_scale * estimate_noise(y)[1], 1.0)
    scale = max(gradient_energy(y), 1e-8)
    for radius in range(min(max_radius, (min(y.shape) - 1) // 2) + 1):
        raw = restore(image, radius, k)
        shown = to_uint8(raw)
        energy = gradient_energy(luminance(shown.astype(np.float32) / 255))
        raw_y = luminance(raw)
        over = np.maximum(-raw_y, 0) + np.maximum(raw_y - 1, 0)
        yield shown, dict(radius=radius, k=k if radius else 0.0, sharpness=energy / (energy + scale),
                          ocr_confidence=ocr_confidence(shown),
                          ringing=min(1.0, float(np.mean(over > 1 / 255) + 4 * np.mean(over))))


def choose(features, settings):
    """Index of the selected candidate; ties (< 1e-4) go to the smaller radius."""
    best, best_score = 0, -float("inf")
    for i, f in enumerate(features):
        s = (f["ocr_confidence"] + settings["sharpness_weight"] * f["sharpness"]
             - settings["ringing_weight"] * f["ringing"])
        if s > best_score + 1e-4:
            best, best_score = i, s
    return best


def restore_adaptive(image, **settings):
    """float RGB in [0, 1] -> restored float RGB in [0, 1]."""
    p = {**DEFAULTS, **settings}
    options = list(candidates(image, p["noise_scale"], p["max_radius"]))
    return options[choose([f for _, f in options], p)][0].astype(np.float32) / 255
