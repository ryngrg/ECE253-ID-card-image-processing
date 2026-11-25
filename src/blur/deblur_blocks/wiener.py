"""Fixed disk Wiener inference, extracted unchanged from the blur experiments."""
import numpy as np
from scipy import fft


def luminance(image):
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def disk_psf(radius):
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (x * x + y * y <= radius * radius).astype(np.float32)
    return kernel / kernel.sum()


def wiener(y, radius, k):
    """Restore luminance fluctuations while preserving the input mean brightness."""
    if type(radius) is not int or radius < 0 or not np.isfinite(k) or k <= 0:
        raise ValueError("Radius must be a nonnegative integer and K must be positive.")
    if radius == 0:
        return y.copy()
    if 2 * radius + 1 > min(y.shape):
        raise ValueError("Blur kernel is larger than the image.")
    pad = 2 * radius
    mean = float(y.mean())
    padded = np.pad(y - mean, pad, mode="reflect")
    kernel = np.zeros(padded.shape, dtype=np.float32)
    psf = disk_psf(radius)
    kernel[: psf.shape[0], : psf.shape[1]] = psf
    kernel = np.roll(kernel, (-radius, -radius), axis=(0, 1))
    h = fft.rfft2(kernel)
    spectrum = fft.rfft2(padded)

    # Blend the outer boundary with a blurred copy to reduce FFT border ringing.
    rows, cols = np.indices(padded.shape, sparse=True)
    distance = np.minimum(
        np.minimum(rows, padded.shape[0] - 1 - rows),
        np.minimum(cols, padded.shape[1] - 1 - cols),
    )
    weight = np.sin(np.minimum(distance / pad, 1) * np.pi / 2) ** 2
    blurred = fft.irfft2(spectrum * h, s=padded.shape)
    spectrum = fft.rfft2(weight * padded + (1 - weight) * blurred)
    restored = fft.irfft2(np.conj(h) * spectrum / (np.abs(h) ** 2 + k), s=padded.shape)
    cropped = restored[pad:-pad, pad:-pad]
    return (cropped - cropped.mean() + mean).astype(np.float32)


def restore(image, radius, k):
    y = luminance(image)
    # Add the same luminance change to all channels, preserving chroma until clipping.
    return image + (wiener(y, radius, k) - y)[..., None]
