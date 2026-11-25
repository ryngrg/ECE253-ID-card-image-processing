#!/usr/bin/env python3
"""Apply one blur-recovery block to a PNG directory, preserving names and dimensions."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from deblur_blocks import DeblurBlock, digest

ROOT = Path(__file__).resolve().parents[1]


def process(in_dir, out_dir, block):
    """Write a complete run to a fresh directory; failed runs leave no partial output."""
    in_dir, out_dir = Path(in_dir).resolve(), Path(out_dir).resolve()
    if not in_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {in_dir}')
    if out_dir.exists() or out_dir == in_dir or in_dir in out_dir.parents:
        raise ValueError('Output must be a new directory outside the input directory.')
    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    other_images = [p.name for p in in_dir.iterdir() if p.is_file()
                    and p.suffix.lower() in {'.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}]
    if other_images:
        raise ValueError('Use PNG inputs for lossless filename-preserving chains; convert other images first.')
    if not files or len({p.name.casefold() for p in files}) != len(files):
        raise ValueError('Expected PNG images with unique filenames (ignoring case).')
    if block.color_order != 'rgb':
        raise ValueError('Directory processing uses RGB; BGR is available through the Python API.')
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve(), *sorted((source / 'deblur_blocks').rglob('*.py')),
               *sorted((source / 'deblur_blocks').rglob('*.json'))]
    run = {'method': block.method, 'settings': block.settings,
           'input_directory': str(in_dir), 'format': 'uint8 RGB PNG',
           'source_sha256': {str(p.relative_to(source)): digest(p) for p in sources},
           'parent_run_sha256': digest(in_dir / 'run.json') if (in_dir / 'run.json').is_file() else None,
           'images': []}
    with tempfile.TemporaryDirectory(prefix='.deblur-', dir=out_dir.parent) as temporary:
        stage = Path(temporary) / 'output'
        stage.mkdir()
        for path in files:
            input_hash = digest(path)
            with Image.open(path) as image:
                if image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError(f'{path.name}: expected a single RGB PNG; convert explicitly first.')
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f'{path.name}: orient pixels and clear EXIF orientation first.')
                original = np.asarray(image).copy()
            start = time.perf_counter()
            result = block(original)
            elapsed = time.perf_counter() - start
            if digest(path) != input_hash:
                raise ValueError(f'Input changed during processing: {path.name}')
            destination = stage / path.name
            Image.fromarray(result).save(destination)
            run['images'].append({'filename': path.name, 'input_sha256': input_hash,
                                  'output_sha256': digest(destination),
                                  'height': original.shape[0], 'width': original.shape[1],
                                  'seconds': elapsed})
            print(f'Processed {path.name}', flush=True)
        (stage / 'run.json').write_text(json.dumps(run, indent=2, allow_nan=False) + '\n')
        if out_dir.exists():
            raise ValueError(f'Output appeared during processing: {out_dir}')
        stage.rename(out_dir)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['wiener', 'wiener_adaptive', 'restormer'])
    parser.add_argument('--in-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--params', type=Path, help='Wiener JSON (radius and k; or the adaptive settings)')
    parser.add_argument('--radius', type=int, help='Override Wiener radius in input pixels; 0 bypasses')
    parser.add_argument('--k', type=float, help='Override fixed Wiener regularization')
    parser.add_argument('--weights', type=Path, help='Restormer checkpoint; default: official pretrained')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=64)
    args = parser.parse_args()
    try:
        settings = {}
        if args.method != 'restormer':
            if args.weights is not None or args.device != 'auto' or args.tile_size != 512 or args.overlap != 64:
                raise ValueError('Weights, device and tiling options apply only to Restormer.')
        if args.method == 'wiener_adaptive':
            if args.radius is not None or args.k is not None:
                raise ValueError('Adaptive Wiener chooses radius and K itself.')
            params = args.params or ROOT / 'deblur_wiener_adaptive_params.json'
            config = json.loads(params.read_text()) if params.is_file() else {}
            settings = {'adaptive': {k: v for k, v in config.items() if k != 'description'}}
        elif args.method == 'wiener':
            params = args.params or ROOT / 'deblur_wiener_params.json'
            config = json.loads(params.read_text())
            settings = {key: config[key] for key in ('radius', 'k')}
            if args.radius is not None:
                settings['radius'] = args.radius
            if args.k is not None:
                settings['k'] = args.k
        else:
            if args.params is not None or args.radius is not None or args.k is not None:
                raise ValueError('Parameters, radius and K apply only to Wiener.')
            settings = dict(weights=args.weights, device=args.device,
                            tile_size=args.tile_size, overlap=args.overlap)
        block = DeblurBlock(args.method, **settings)
        run = process(args.in_dir, args.out_dir, block)
        print(f'Saved {len(run["images"])} images and run.json to {args.out_dir}')
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
