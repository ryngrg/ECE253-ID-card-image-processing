"""Trainer, adaptive Wiener and selection checks; no datasets, OCR engines or model download required."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from deblur_blocks import adaptive_wiener as aw


def test_noise_estimate_recovers_gaussian_sigma():
    rng = np.random.default_rng(0)
    y = 0.5 + rng.normal(0, 0.02, (201, 303))
    assert aw.estimate_noise(y)[0] == pytest.approx(0.02, rel=0.05)


def test_adaptive_choice_prefers_confidence_and_breaks_ties_to_smaller_radius():
    f = [dict(ocr_confidence=c, sharpness=0.5, ringing=0.0) for c in (0.5, 0.8, 0.8)]
    assert aw.choose(f, aw.DEFAULTS) == 1
    f[1]['ringing'] = 0.5
    assert aw.choose(f, aw.DEFAULTS) == 2


def test_adaptive_block_keeps_shape_and_rejects_unknown_settings(monkeypatch):
    from deblur_blocks import DeblurBlock
    monkeypatch.setattr(aw, 'ocr_confidence', lambda image: 0.5)
    image = np.random.default_rng(1).integers(0, 256, (40, 61, 3), dtype=np.uint8)
    out = DeblurBlock('wiener_adaptive', adaptive={'max_radius': 2})(image)
    np.testing.assert_array_equal(out, image)  # equal confidence, ringing penalised -> radius 0
    with pytest.raises(ValueError):
        DeblurBlock('wiener_adaptive', adaptive={'radius': 2})


def test_selection_rule():
    pytest.importorskip('skimage')
    from tune_deblur import select
    t = pd.DataFrame(dict(char_acc=[.5, .9, .6, .7], ssim=[.5, .4, .6, .55]))
    assert select(t) == (3, True)        # row 1 is best by accuracy but loses SSIM
    t = pd.DataFrame(dict(char_acc=[.5, .9, .4], ssim=[.5, .4, .3]))
    assert select(t) == (1, False)


def test_tone_match_removes_gain_but_not_sharpness():
    pytest.importorskip('torch')
    from finetune_deblur import match_tone
    import cv2
    sharp = np.random.default_rng(2).random((160, 160, 3)).astype(np.float32)
    sharp = cv2.GaussianBlur(sharp, (0, 0), 1) * 0.5 + np.linspace(0, .4, 160, dtype=np.float32)[None, :, None]
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3) * 0.8 + 0.05
    out = match_tone(sharp, blurred)
    np.testing.assert_allclose(out, sharp * 0.8 + 0.05, atol=0.02)


def test_training_never_accepts_test_rows(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from finetune_deblur import load_pairs
    monkeypatch.chdir(tmp_path)
    pd.DataFrame(dict(card_id=['A', 'B'], split=['train', 'test'])).to_csv('split.csv', index=False)
    pd.DataFrame(dict(card_id=['A', 'B'], split=['train', 'train'], usable=True, path='x')).to_csv('p.csv', index=False)
    with pytest.raises(ValueError, match='disagrees'):
        load_pairs('p.csv', 'train')
    with pytest.raises(ValueError, match='train and val'):
        load_pairs('p.csv', 'test')


def test_patches_stay_inside_and_pair_input_with_target():
    pytest.importorskip('torch')
    from finetune_deblur import training_patch
    x = np.random.default_rng(3).random((90, 120, 3)).astype(np.float32)
    s = [dict(card_id='A', input=x, target=x + 1, centers=np.array([[2., 2.], [118., 88.]]))]
    rng = np.random.default_rng(4)
    for _ in range(20):
        a, b = training_patch(s, 64, rng, 0.75)
        assert a.shape == (64, 64, 3)
        np.testing.assert_allclose(b, a + 1)


def test_directory_run_can_be_limited_to_named_files(tmp_path):
    from PIL import Image
    from deblur import process
    from deblur_blocks import DeblurBlock
    (tmp_path / 'in').mkdir()
    for n in 'abc':
        Image.fromarray(np.full((20, 24, 3), 100, np.uint8)).save(tmp_path / 'in' / f'{n}.png')
    block = DeblurBlock('wiener', radius=1)
    process(tmp_path / 'in', tmp_path / 'out', block, {'a.png', 'c.png'})
    assert sorted(p.name for p in (tmp_path / 'out').glob('*.png')) == ['a.png', 'c.png']
    with pytest.raises(ValueError, match='Missing'):
        process(tmp_path / 'in', tmp_path / 'out2', block, {'z.png'})
