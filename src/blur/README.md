# Deblurring blocks (`src/blur/`)

Wiener (fixed and adaptive) and Restormer as independent, chainable blocks: directory of PNGs in, directory of PNGs
out, same filenames and size. They do no OCR, alignment or evaluation, so they can run on any rectified image folder
or on the output of another block (glare removal included). Results and the method summary are in the main
[README](../../README.md). Run everything from the repository root.

| file | what it does |
|---|---|
| `deblur.py` | CLI: `--method wiener \| wiener_adaptive \| restormer`, `--in-dir`, `--out-dir` |
| `deblur_blocks/` | the blocks (`wiener.py`, `adaptive_wiener.py`, `restormer.py`); Restormer architecture + MIT license vendored in `vendor/` |
| `install_deblur_weights.py` | downloads / installs Restormer weights with size and SHA-256 checks (`deblur_blocks/models.json`) |
| `tune_deblur.py` | Wiener grid, adaptive Wiener settings and Restormer step selection on our train + val |
| `finetune_deblur.py` | fine-tunes Restormer on our train B_D pairs |

## Install
`pip install -r requirements.txt` (repository root) covers everything. Wiener only needs NumPy, SciPy and Pillow
and imports neither PyTorch nor an OCR engine; the adaptive Wiener also calls Tesseract.
```bash
python src/blur/install_deblur_weights.py --model pretrained               # official defocus weights (~100 MB)
python src/blur/install_deblur_weights.py --model ours --from PATH/restormer_ours.pt   # our fine-tuned model
```
`checkpoints/blur/restormer_ours.pt` (git-ignored) is step 1000 of `finetune_deblur.py`, selected on val, SHA-256
`a6488d0fbcfadcbb583a0eff1e157b7cb5ccac6733223038f25422b66c263e4d`. Every result in the README uses it.

## Run a block
```bash
python src/blur/deblur.py --method wiener          --in-dir data/rectified --out-dir out/wiener
python src/blur/deblur.py --method wiener_adaptive --in-dir data/rectified --out-dir out/wiener_adaptive  # ~3 s/image
python src/blur/deblur.py --method restormer --weights checkpoints/blur/restormer_ours.pt \
  --in-dir data/rectified --out-dir out/restormer                                                         # ~20 s/image (MPS)
python src/blur/deblur.py --method restormer --in-dir out/classical --out-dir out/classical_restormer     # chain: input = another block's output
```
* Wiener defaults to `params/blur/wiener_params.json` (disk radius 6, K 0.03, in pixels of the rectified cards).
  `--radius R --k K` overrides it; `--radius 0` passes the image through.
* Restormer without `--weights` uses the official pretrained weights. Tiles are 512 px with 64 px overlap
  (`--tile-size`, `--overlap`, multiples of 8); one tile at a time is on the device. `--device auto` picks CUDA,
  then MPS, then CPU. No resizing.
* `--pairs FILE` restricts the run to the images of a pairs table (as `run_chain_experiments.sh` does).

## Tune and train on our splits
```bash
python src/common/prepare_data.py                      # out/pairs_blur.csv (B_D -> S_D; 16 / 4 / 4)
python src/blur/tune_deblur.py wiener --write          # radius x K grid on train + val, ~1 h
python src/blur/tune_deblur.py wiener_adaptive --write # train + val
python src/blur/finetune_deblur.py                     # 2000 steps on train -> checkpoints/blur/restormer_ours/
python src/blur/tune_deblur.py restormer --write       # picks a step on val -> checkpoints/blur/restormer_ours.pt
```
Selection rule: among the candidates that raise both character accuracy (fixed boxes, mean of both engines) and SSIM
over the untreated images, the one with the highest accuracy. Logs of the selection: `results/blur/tune_deblur/`.
Test cards are never loaded: the pairs file must agree with `split.csv`, and `tune_deblur.py restormer` refuses a
checkpoint whose recorded training cards are not all train cards.

## Image and directory contract
* Inputs: single-frame, already oriented **8-bit RGB PNGs** (rectified images and every block's output qualify).
  JPEG / TIFF and other non-PNG images are rejected, not skipped or recompressed. No recursion; non-image files
  such as `run.json` are ignored.
* Outputs keep the exact filename (including `_rectified.png`), width, height and pixel grid; values are rounded and
  clipped to uint8 and saved losslessly, without the source metadata.
* The output directory must not exist and must lie outside the input directory. A failed batch leaves no partial
  output; inputs are never modified.
* `run.json` in the output records the parameters, model hash, per-image input / output hashes and sizes, processing
  times (without model loading and disk I/O) and the hash of the previous block's `run.json`, if any.

## Python interface
With `src/` on `sys.path` (as in every script under `src/`):
```python
from blur.deblur_blocks import DeblurBlock

wiener = DeblurBlock('wiener', radius=3, k=0.3, color_order='bgr')
restormer = DeblurBlock('restormer', color_order='bgr')
restored = restormer(wiener(image))   # image: OpenCV uint8 HxWx3 BGR array
```
The default color order is `rgb` (Pillow / NumPy); set `color_order='bgr'` for OpenCV, channels are not guessed.
Each block returns uint8 in the input's color order and shape and does not modify its input; between blocks the
values are rounded to uint8, as in a PNG directory chain. Create a block once and reuse it (the model loads once).

## Provenance
Wiener and Restormer inference were taken unchanged from the teammate's blur repository
(`deblur_blocks/provenance.json`); training, tuning and the adaptive Wiener are adapted ports. Earlier blur
experiments, and why their models are not used: `docs/blur_history/README.md`.

## Tests
```bash
python -m pytest -q tests/blur
```
No data or downloaded model needed: image contract, directory chaining, color order, tiling, weight hash checks,
cleanup after a failure, loading a block once, and the training / tuning helpers. PyTorch tests skip without torch.
