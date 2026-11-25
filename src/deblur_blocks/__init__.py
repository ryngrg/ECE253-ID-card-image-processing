"""Chainable image-to-image blocks. Public arrays are uint8 HxWx3 RGB or BGR."""
import hashlib
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class DeblurBlock:
    """Load once, call repeatedly; never modify the caller's image array.

    color_order='bgr' supports cv2.imread/imwrite directly. RGB is the default.
    Every call returns uint8 with the same shape and color order as its input.
    """

    def __init__(self, method, *, radius=3, k=0.3, weights=None,
                 device='auto', tile_size=512, overlap=64, color_order='rgb', adaptive=None):
        if color_order not in {'rgb', 'bgr'}:
            raise ValueError('color_order must be rgb or bgr')
        self.color_order = color_order
        self.method = method
        if method == 'wiener':
            if type(radius) is not int or radius < 0 or not np.isfinite(k) or k <= 0:
                raise ValueError('Radius must be a nonnegative integer and K positive and finite.')
            from .wiener import restore
            self.apply = lambda image: restore(image, radius, k)
            self.settings = {'radius': radius, 'k': k}
        elif method == 'wiener_adaptive':
            from .adaptive_wiener import DEFAULTS, restore_adaptive
            self.settings = {**DEFAULTS, **(adaptive or {})}
            if set(self.settings) != set(DEFAULTS):
                raise ValueError(f'Adaptive Wiener settings are {sorted(DEFAULTS)}.')
            self.apply = lambda image: restore_adaptive(image, **self.settings)
        elif method == 'restormer':
            if (tile_size < 8 or tile_size % 8 or overlap < 0
                    or overlap >= tile_size or overlap % 8):
                raise ValueError('Tiles/overlap must be multiples of 8, with 0 <= overlap < tile size.')
            from .restormer import DEFAULT_WEIGHTS, RestormerFilter
            weights = Path(weights) if weights is not None else DEFAULT_WEIGHTS
            if not weights.is_file():
                raise FileNotFoundError(
                    f'Weights missing: {weights}. Run python src/install_deblur_weights.py '
                    '--model pretrained, or pass --weights PATH.')
            self.apply = RestormerFilter(weights, device, tile_size, overlap)
            self.settings = {'weights': weights.name,
                             'weights_sha256': self.apply.provenance['weights_sha256'],
                             'device': str(self.apply.device),
                             'tile_size': tile_size, 'overlap': overlap,
                             'task': self.apply.provenance['task'],
                             'fine_tuned': self.apply.provenance['fine_tuning'] is not None}
        else:
            raise ValueError('method must be wiener, wiener_adaptive or restormer')

    def __call__(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError('Expected a nonempty uint8 HxWx3 image.')
        rgb = image[..., ::-1] if self.color_order == 'bgr' else image
        result = np.asarray(self.apply(rgb.astype(np.float32) / 255))
        if result.shape != image.shape or not np.isfinite(result).all():
            raise ValueError('Filter returned invalid dimensions or nonfinite pixels.')
        result = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(result[..., ::-1] if self.color_order == 'bgr' else result)
