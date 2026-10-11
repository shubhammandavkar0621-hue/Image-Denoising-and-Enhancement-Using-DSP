"""2-D FFT and frequency-domain filters for RGB images."""
import numpy as np


def fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Return log-scaled, centered magnitude spectrum for visualization."""
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 3:
        # Average channels for a compact spectrum visualization.
        arr = arr.mean(axis=2)
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.log1p(np.abs(spectrum))
    lo, hi = float(magnitude.min()), float(magnitude.max())
    if hi <= lo:
        return np.zeros_like(magnitude, dtype=np.uint8)
    return np.round((magnitude - lo) / (hi - lo) * 255).astype(np.uint8)


def frequency_filter(
    image: np.ndarray,
    filter_type: str = "Butterworth",
    cutoff: float = 0.12,
    order: int = 2,
) -> np.ndarray:
    """Apply an ideal, Butterworth, or Gaussian low-pass filter in the 2-D FFT domain.

    cutoff is a normalized radius from 0.01 to 0.5, relative to the smaller image dimension.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or RGB image")

    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    d0 = max(1.0, float(cutoff) * min(h, w))
    kind = filter_type.lower()

    if kind == "ideal":
        mask = (distance <= d0).astype(np.float32)
    elif kind == "butterworth":
        n = max(1, int(order))
        mask = 1.0 / (1.0 + (distance / d0) ** (2 * n))
    elif kind == "gaussian":
        mask = np.exp(-(distance ** 2) / (2.0 * d0 ** 2))
    else:
        raise ValueError(f"Unknown frequency filter: {filter_type}")

    def process_channel(channel: np.ndarray) -> np.ndarray:
        shifted = np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))
        filtered = shifted * mask
        restored = np.fft.ifft2(np.fft.ifftshift(filtered))
        return np.real(restored)

    if img.ndim == 2:
        out = process_channel(img)
    else:
        out = np.stack([process_channel(img[..., c]) for c in range(img.shape[2])], axis=-1)

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
