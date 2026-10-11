"""Full-reference image quality metrics."""
import numpy as np
from skimage.metrics import structural_similarity


def _uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return arr


def calculate_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Calculate MSE, PSNR, and SSIM between equally sized reference and estimate images."""
    ref = _uint8(reference)
    est = _uint8(estimate)
    if ref.shape != est.shape:
        raise ValueError(f"Reference and estimate shapes differ: {ref.shape} vs {est.shape}")

    diff = ref.astype(np.float32) - est.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    if ref.ndim == 3:
        ssim = float(structural_similarity(ref, est, channel_axis=-1, data_range=255))
    else:
        ssim = float(structural_similarity(ref, est, data_range=255))
    return {"MSE": mse, "PSNR (dB)": psnr, "SSIM": ssim}
