#!/usr/bin/env python3
"""Download official weights or install a supplied checkpoint with size/SHA-256 verification."""
import argparse
import json
import sys
from pathlib import Path
import shutil
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from blur.deblur_blocks import digest

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).resolve().parent / 'deblur_blocks/models.json'


def install(model, from_path=None, output_dir=None):
    record = json.loads(CATALOG.read_text())[model]
    folder = Path(output_dir) if output_dir is not None else ROOT / 'checkpoints/blur'
    target = folder / record['filename']

    def verify(path):
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Checkpoint failed size/SHA-256 verification; destination unchanged.')

    if target.exists():
        verify(target)
        return target
    if from_path is None and not record.get('source'):
        raise ValueError(f'The {model} checkpoint is distributed separately. Use --from PATH; see src/blur/README.md.')
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.weights-', dir=folder) as temporary:
        staged = Path(temporary) / record['filename']
        if from_path is not None:
            shutil.copyfile(from_path, staged)
        else:
            with urllib.request.urlopen(record['source'], timeout=60) as response, staged.open('wb') as output:
                shutil.copyfileobj(response, output)
        verify(staged)
        # Exclusive destination creation prevents replacing weights from another run.
        try:
            with target.open('xb') as output, staged.open('rb') as source:
                shutil.copyfileobj(source, output)
        except FileExistsError:
            verify(target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pretrained', 'ours'], required=True)
    parser.add_argument('--from', dest='from_path', type=Path, help='Copy local weights instead of downloading')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        print('Verified:', install(args.model, args.from_path, args.output_dir))
    except (ValueError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
