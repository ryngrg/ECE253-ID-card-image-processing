"""Official single-image defocus Restormer with native-resolution tiled inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .vendor.restormer_arch import Restormer

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
DEFAULT_WEIGHTS = ROOT / 'checkpoints/deblur/single_image_defocus_deblurring.pth'
ARCHITECTURE = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                    num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                    bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
TASK = 'single_image_defocus_deblurring'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def unit_rgb(image):
    a = np.asarray(image)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255
    elif np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
    else:
        raise ValueError('Use uint8 RGB or floating RGB in [0,1].')
    if a.ndim != 3 or a.shape[2] != 3 or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite HxWx3 RGB.')
    if a.min() < 0 or a.max() > 1:
        raise ValueError('RGB values must lie in [0,1].')
    return a


def to_tensor(image, device):
    return torch.from_numpy(np.ascontiguousarray(unit_rgb(image).transpose(2, 0, 1))).unsqueeze(0).to(device)


def to_image(image):
    return Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8))


def select_device(value='auto'):
    if value != 'auto':
        return torch.device(value)
    return torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')


def load_model(weights=DEFAULT_WEIGHTS, device='cpu'):
    """Load the official checkpoint or a checkpoint produced by this trainer, strictly."""
    weights = Path(weights)
    source = PACKAGE / 'vendor'
    provenance = json.loads((source / 'provenance.json').read_text())
    expected_arch = provenance['files']['restormer_arch.py']['sha256']
    if digest(source / 'restormer_arch.py') != expected_arch:
        raise ValueError('Vendored upstream architecture has changed.')
    weights_hash = digest(weights)
    original = json.loads((PACKAGE / 'models.json').read_text())['pretrained']
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('restormer_experiment')
    if metadata is None:
        if weights_hash != original['sha256']:
            raise ValueError('Unrecognized checkpoint: expected the official single-image defocus weights.')
    elif (metadata.get('task') != TASK or metadata.get('architecture') != ARCHITECTURE
          or metadata.get('pretrained_sha256') != original['sha256']
          or metadata.get('architecture_sha256') != expected_arch):
        raise ValueError('Fine-tuned checkpoint has incompatible task/architecture/provenance.')
    model = Restormer(**ARCHITECTURE)
    model.load_state_dict(checkpoint['params'], strict=True)
    model.to(device).eval()
    return model, {'task': TASK, 'weights_sha256': weights_hash,
                   'pretrained_sha256': original['sha256'], 'architecture_sha256': expected_arch,
                   'architecture': ARCHITECTURE, 'fine_tuning': metadata,
                   'upstream_commit': provenance['commit']}


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, tile - overlap))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


@torch.inference_mode()
def tiled_inference(model, image, device='cpu', tile_size=256, overlap=32):
    """Uniformly average overlaps, as in the official demo, without resizing.

    CPU holds the full image and accumulators; only one tile enters the model.
    Restormer's attention depends on tile content, so this is not mathematically
    equivalent to full-image inference. Use identical tiling for all comparisons.
    """
    if tile_size < 8 or tile_size % 8 or overlap < 0 or overlap >= tile_size or overlap % 8:
        raise ValueError('Tile size and overlap must be multiples of 8, with 0 <= overlap < tile size.')
    original = unit_rgb(image)
    height, width = original.shape[:2]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    # Numpy reflection also handles dimensions smaller than the padding length.
    padded = np.pad(original, ((0, pad_h), (0, pad_w), (0, 0)),
                    mode='reflect' if min(height, width) > 1 else 'edge')
    h, w = padded.shape[:2]
    total = np.zeros_like(padded)
    counts = np.zeros((h, w, 1), dtype=np.float32)
    model.eval()
    for y in tile_starts(h, tile_size, overlap):
        for x in tile_starts(w, tile_size, overlap):
            patch = padded[y:y+tile_size, x:x+tile_size]
            prediction = model(to_tensor(patch, device))
            if prediction.shape != (1, 3, *patch.shape[:2]) or not torch.isfinite(prediction).all():
                raise RuntimeError('Invalid Restormer output shape or nonfinite prediction.')
            array = prediction[0].permute(1, 2, 0).float().cpu().numpy()
            ph, pw = patch.shape[:2]
            total[y:y+ph, x:x+pw] += array
            counts[y:y+ph, x:x+pw] += 1
    if np.any(counts == 0):
        raise RuntimeError('Tiling left uncovered pixels.')
    return np.clip(total / counts, 0, 1)[:height, :width]


class RestormerFilter:
    def __init__(self, weights=DEFAULT_WEIGHTS, device='auto', tile_size=256, overlap=32):
        self.device = select_device(device)
        self.model, self.provenance = load_model(weights, self.device)
        self.tile_size, self.overlap = tile_size, overlap

    def __call__(self, image):
        return tiled_inference(self.model, image, self.device, self.tile_size, self.overlap)
