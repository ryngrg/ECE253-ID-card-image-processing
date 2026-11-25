# Chainable blur-recovery blocks

Run scripts from the repository root. These blocks apply the existing fixed
Wiener filter or single-image defocus Restormer to images without OCR, reference
images, training data, alignment, or metric evaluation. They can process an
original directory or the output directory of another restoration block.

## Install

```bash
python -m pip install -r requirements-deblur.txt
# Only needed for Restormer:
python -m pip install -r requirements-deblur-restormer.txt
python src/blur/install_deblur_weights.py --model pretrained
```

The installer downloads the authors' official defocus checkpoint and verifies
its recorded size and SHA-256. The architecture and MIT license are vendored
under `src/blur/deblur_blocks/vendor/`; original source provenance is recorded there.
Wiener needs only NumPy, SciPy and Pillow; it does not import PyTorch or OCR.
Use Python 3.10 or newer.

## Run a block

```bash
python src/blur/deblur.py --method wiener \
  --in-dir data/rectified --out-dir out/wiener

python src/blur/deblur.py --method restormer \
  --in-dir data/rectified --out-dir out/restormer_pretrained

# Override fixed Wiener settings or supply another radius/K JSON:
python src/blur/deblur.py --method wiener --radius 2 --k 0.1 \
  --in-dir data/rectified --out-dir out/wiener_r2

# Process an existing block's output, preserving its filenames:
python src/blur/deblur.py --method restormer \
  --in-dir out/classical --out-dir out/classical_then_restormer

# Adaptive Wiener: radius and noise level estimated per image (needs Tesseract, ~3 s/image):
python src/blur/deblur.py --method wiener_adaptive \
  --in-dir data/rectified --out-dir out/wiener_adaptive
```

Restormer takes ~20 s per rectified card on an M2 (MPS).

`--params params/blur/wiener_params.json` is the default for Wiener: radius 6 / K 0.03, chosen on our train + val B_D
(`tune_deblur.py`), in pixels of the rectified cards. `--radius 0` bypasses Wiener.

Restormer defaults to the **official pretrained** weights. `--weights PATH`
selects a compatible fine-tuned checkpoint explicitly. It loads once per run.
Default tiles are 512 pixels with overlap 64; `--tile-size` and `--overlap` must
be multiples of 8, with overlap smaller than the tile. `--device auto` chooses
CUDA, then MPS, then CPU; use `--device cpu|mps|cuda` to select explicitly.
Only a tile is placed on the device at a time. No resizing is performed.

Chaining support describes the interface, not evidence that a particular order
improves restoration.

## Tune and train on our splits

```bash
python src/common/prepare_data.py                       # writes out/pairs_blur.csv (B_D -> S_D; 16 / 4 / 4)
python src/blur/tune_deblur.py wiener --write          # radius x K grid on train + val, ~1 h
python src/blur/tune_deblur.py wiener_adaptive --write # train + val
python src/blur/finetune_deblur.py                     # 2000 steps on train -> checkpoints/blur/restormer_ours/
python src/blur/tune_deblur.py restormer --write       # picks a step on val -> checkpoints/blur/restormer_ours.pt
python src/blur/deblur.py --method restormer --weights checkpoints/blur/restormer_ours.pt \
  --in-dir data/rectified --out-dir out/restormer_ours
```

Test cards are never loaded: the pairs file must agree with `split.csv`, and `tune_deblur.py restormer` refuses a
checkpoint whose recorded training cards are not all train cards. Selection rule: among the candidates
that improve both character accuracy (fixed boxes, mean of both engines) and SSIM over the untreated images, the
highest accuracy. Earlier experiments and their results: `docs/blur_history/`.

## Image and directory contract

- Inputs are single-frame, already oriented **8-bit RGB PNGs**. Existing
  rectified RGB PNGs and these blocks' outputs satisfy this contract.
- Every output keeps the exact filename (including `_rectified.png`), width,
  height, and pixel coordinate system. Results are clipped/rounded to uint8 and
  saved losslessly; embedded source metadata is not copied.
- No recursion. Nonimage files such as `run.json` are ignored. JPEG/TIFF and
  other recognized non-PNG images are rejected instead of silently omitted or
  recompressed. Convert non-RGB or EXIF-oriented inputs explicitly beforehand.
- The output directory must not exist and must be outside the input directory.
  A failed batch leaves no partial output directory; inputs are never modified.
- `run.json` records effective parameters, model hash, source hashes, per-image
  input/output hashes and dimensions, processing times (excluding model load and
  disk I/O), and the previous block's `run.json` hash when present.

Output directories can be supplied to the glare repository's existing
`evaluate.py --treated-dir` or `box_ocr.py --extra DIR:NAME` when evaluation is
wanted later. Those scripts are not called by a blur block.

## Python / OpenCV interface

With `src/` on `sys.path` (as in every script under `src/`):

```python
from blur.deblur_blocks import DeblurBlock

wiener = DeblurBlock('wiener', radius=3, k=0.3, color_order='bgr')
restormer = DeblurBlock('restormer', color_order='bgr')
# image is an OpenCV uint8 HxWx3 BGR array.
restored = restormer(wiener(image))
```

The default color order is `rgb` for Pillow/NumPy callers. Every block returns
uint8 in the input color order with the same shape, and does not mutate the
input. Set `color_order='bgr'` explicitly for OpenCV; channels are not guessed.
At each block boundary, outputs are clipped/rounded to uint8 just as in a PNG
directory chain. Instantiate each block once and reuse it across images.

## Our model

`checkpoints/blur/restormer_ours.pt` (git-ignored): step 1000 of `finetune_deblur.py`, selected on val, SHA-256
`a6488d0fbcfadcbb583a0eff1e157b7cb5ccac6733223038f25422b66c263e4d`.
All results in README use it; install a copy with `python src/blur/install_deblur_weights.py --model ours --from PATH`.

## Pilot model (do not use for results)

The earlier fine-tuned model (`real-pilot`, `step_002000.pt`, SHA-256
`1f9da934b848c418137c023be02ac3dd5b5815ed7716ed4b9d30dc01988eac63`) was trained on F03 (our test), F01 (our val) and
MIDV frames, one of which is from an external test clip. `docs/blur_history/README.md` has the full overlap. It can
still be installed for inspection:

```bash
python src/blur/install_deblur_weights.py --model real-pilot --from /path/to/step_002000.pt
```

Wiener and Restormer inference was extracted unchanged from the blur repo (`src/blur/deblur_blocks/provenance.json`);
training, tuning and the adaptive Wiener are adapted ports (`docs/blur_history/README.md`).

## Checks

```bash
python -m pip install "pytest<9"   # 9.2.0 crashes on startup in our Python 3.10 venv
python -m pytest -q tests/blur
```

These checks require no datasets or downloaded model. They verify the image
contract, CLI directory chaining, color order, tiling, installation hash checks,
failure cleanup, and loading a block once. PyTorch checks skip if unavailable.
The original blur repository also has tests against its legacy implementation.
