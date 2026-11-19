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
python src/install_deblur_weights.py --model pretrained
```

The installer downloads the authors' official defocus checkpoint and verifies
its recorded size and SHA-256. The architecture and MIT license are vendored
under `src/deblur_blocks/vendor/`; original source provenance is recorded there.
Wiener needs only NumPy, SciPy and Pillow; it does not import PyTorch or OCR.
Use Python 3.10 or newer.

## Run a block

```bash
python src/deblur.py --method wiener \
  --in-dir data/rectified --out-dir out/wiener

python src/deblur.py --method restormer \
  --in-dir data/rectified --out-dir out/restormer_pretrained

# Override fixed Wiener settings or supply another radius/K JSON:
python src/deblur.py --method wiener --radius 2 --k 0.1 \
  --in-dir data/rectified --out-dir out/wiener_r2

# Process an existing block's output, preserving its filenames:
python src/deblur.py --method restormer \
  --in-dir out/classical --out-dir out/classical_then_restormer
```

`--params deblur_wiener_params.json` is the default for Wiener. It carries the
existing real-data pilot setting, radius 3 / K 0.3, without denoising or
sharpening. Radius is measured in input pixels. These settings have not been
retuned for resized or rectified cards. `--radius 0` bypasses Wiener.

Restormer defaults to the **official pretrained** weights. `--weights PATH`
selects a compatible fine-tuned checkpoint explicitly. It loads once per run.
Default tiles are 512 pixels with overlap 64; `--tile-size` and `--overlap` must
be multiples of 8, with overlap smaller than the tile. `--device auto` chooses
CUDA, then MPS, then CPU; use `--device cpu|mps|cuda` to select explicitly.
Only a tile is placed on the device at a time. No resizing is performed.

Chaining support describes the interface, not evidence that a particular order
improves restoration. No joint experiment or retraining is performed here.

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

When importing from a script under `src/`:

```python
from deblur_blocks import DeblurBlock

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

## Existing fine-tuned model

The blur workspace's selected real-data checkpoint is
`results/real_mixed_pilot/restormer/step_002000.pt`. Weights are distributed
separately from source; copying this file from that workspace is sufficient.
Its SHA-256 is
`1f9da934b848c418137c023be02ac3dd5b5815ed7716ed4b9d30dc01988eac63`.

```bash
python src/install_deblur_weights.py --model real-pilot \
  --from /path/to/step_002000.pt
python src/deblur.py --method restormer \
  --weights checkpoints/deblur/restormer_real_pilot_step_002000.pt \
  --in-dir data/rectified --out-dir out/restormer_real_pilot
```

The installer verifies the exact file; `--weights` also accepts compatible
checkpoints from the original blur trainer with matching architecture/task
metadata. The pilot trained on handmade F03, which is in the glare repository's
test split. Its output is available for integration, but it is **not** an unseen
model on that full test split. The transferred Wiener setting was selected in
that same blur experiment. This package does not change either dataset split.

Training/evaluation code and historical results remain in the original private
blur repository, `https://github.com/Mlazazarus/ECE253`. The portable package is
a documented snapshot: Wiener functions are extracted unchanged; Restormer
inference has adapted imports/asset paths. Source hashes are in
`src/deblur_blocks/provenance.json`. Future inference changes should rerun the
local legacy-equivalence checks before exporting again.

## Checks

```bash
python -m pip install pytest
python -m pytest -q tests/deblur
```

These checks require no datasets or downloaded model. They verify the image
contract, CLI directory chaining, color order, tiling, installation hash checks,
failure cleanup, and loading a block once. PyTorch checks skip if unavailable.
The original blur repository also has tests against its legacy implementation.
