"""Portable contract checks; no datasets, OCR engines or model download required."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from blur.deblur_blocks import DeblurBlock, digest
from blur.deblur import process
from blur import install_deblur_weights as weights


def picture():
    return np.random.default_rng(123).integers(0, 256, (29, 43, 3), dtype=np.uint8)


def test_color_order_and_no_input_mutation():
    rgb = picture()
    before = rgb.copy()
    actual = DeblurBlock('wiener')(rgb)
    bgr = DeblurBlock('wiener', color_order='bgr')(rgb[..., ::-1])
    np.testing.assert_array_equal(bgr[..., ::-1], actual)
    np.testing.assert_array_equal(rgb, before)
    assert actual.dtype == np.uint8 and actual.shape == rgb.shape


@pytest.mark.parametrize('bad', [np.zeros((12, 12)), np.zeros((12, 12, 3)),
                                 np.zeros((0, 12, 3), dtype=np.uint8)])
def test_explicit_image_contract(bad):
    with pytest.raises(ValueError, match='uint8'):
        DeblurBlock('wiener')(bad)


def test_directory_chaining_preserves_pixels_names_and_parent_provenance(tmp_path):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    image = picture()
    filename = 'F99_B_G_0_rectified.png'
    Image.fromarray(image).save(inputs / filename)
    # Invoke the actual CLI from outside the repo to catch path/import problems.
    for first, second in [(inputs, tmp_path / 'a'), (tmp_path / 'a', tmp_path / 'b')]:
        subprocess.run([sys.executable, str(ROOT / 'src/blur/deblur.py'), '--method', 'wiener',
                        '--radius', '0', '--in-dir', str(first), '--out-dir', str(second)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / 'b' / filename)), image)
    run = json.loads((tmp_path / 'b/run.json').read_text())
    assert run['parent_run_sha256'] == digest(tmp_path / 'a/run.json')
    assert run['images'][0]['output_sha256'] == digest(tmp_path / 'b' / filename)
    assert run['images'][0]['width'] == image.shape[1]


def test_failed_batch_leaves_inputs_and_destination_untouched(tmp_path):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    Image.fromarray(picture()).save(inputs / 'a.png')
    (inputs / 'z.png').write_bytes(b'not an image')
    before = digest(inputs / 'a.png')
    output = tmp_path / 'output'
    with pytest.raises(OSError):
        process(inputs, output, DeblurBlock('wiener'))
    assert not output.exists()
    assert digest(inputs / 'a.png') == before
    assert not list(tmp_path.glob('.deblur-*'))


def test_no_overwrite_or_nested_output(tmp_path):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    Image.fromarray(picture()).save(inputs / 'a.png')
    for output in (inputs, inputs / 'nested'):
        with pytest.raises(ValueError, match='new directory'):
            process(inputs, output, DeblurBlock('wiener'))


def test_lossy_inputs_are_not_silently_skipped(tmp_path):
    Image.fromarray(picture()).save(tmp_path / 'image.jpg')
    with pytest.raises(ValueError, match='PNG inputs'):
        process(tmp_path, tmp_path.parent / 'never-written', DeblurBlock('wiener'))


def test_tiling_preserves_odd_and_tiny_grids():
    torch = pytest.importorskip('torch')
    pytest.importorskip('einops')
    from blur.deblur_blocks.restormer import tiled_inference
    for shape in [(1, 1, 3), (3, 11, 3), (29, 43, 3)]:
        image = np.random.default_rng(1).random(shape, dtype=np.float32)
        result = tiled_inference(torch.nn.Identity(), image, tile_size=16, overlap=8)
        np.testing.assert_allclose(result, image, atol=2e-7)


def test_restormer_loads_once_and_chain_keeps_contract(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('einops')
    from blur.deblur_blocks import restormer
    calls = []

    class FakeFilter:
        def __init__(self, *args):
            calls.append(args)
            self.device = 'cpu'
            self.provenance = {'weights_sha256': 'test', 'task': restormer.TASK, 'fine_tuning': None}

        def __call__(self, image):
            return image

    monkeypatch.setattr(restormer, 'RestormerFilter', FakeFilter)
    block = DeblurBlock('restormer', weights=Path(__file__))
    for _ in range(2):
        original = picture()
        np.testing.assert_array_equal(block(DeblurBlock('wiener', radius=0)(original)), original)
    assert len(calls) == 1


def test_checkpoint_install_verifies_before_writing(tmp_path, monkeypatch):
    supplied = tmp_path / 'supplied.pt'
    supplied.write_bytes(b'test checkpoint')
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps({'fixture': {'filename': 'weights.pt', 'bytes': supplied.stat().st_size,
                                              'sha256': digest(supplied), 'source': None}}))
    monkeypatch.setattr(weights, 'CATALOG', catalog)
    output = weights.install('fixture', supplied, tmp_path / 'installed')
    assert output.read_bytes() == supplied.read_bytes()
    output.write_bytes(b'different existing weights')
    with pytest.raises(ValueError, match='verification'):
        weights.install('fixture', supplied, output.parent)
    assert output.read_bytes() == b'different existing weights'
    supplied.write_bytes(b'invalid')
    with pytest.raises(ValueError, match='verification'):
        weights.install('fixture', supplied, tmp_path / 'bad')
    assert not (tmp_path / 'bad/weights.pt').exists()


def test_wiener_does_not_import_torch_or_ocr():
    code = ("import sys; from blur.deblur_blocks import DeblurBlock; DeblurBlock('wiener'); "
            "assert not {'torch', 'pytesseract', 'OCR', 'pywt'} & sys.modules.keys()")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'src', check=True)
