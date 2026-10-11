"""Spatial-domain denoising filters."""
import cv2
import numpy as np


def _odd_kernel(kernel_size: int) -> int:
    k = max(3, int(kernel_size))
    return k if k % 2 else k + 1


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = _odd_kernel(kernel_size)
    result = cv2.blur(image, (k, k), borderType=cv2.BORDER_REFLECT)
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 0.0) -> np.ndarray:
    k = _odd_kernel(kernel_size)
    result = cv2.GaussianBlur(image, (k, k), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)
    return np.clip(result, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = _odd_kernel(kernel_size)
    result = cv2.medianBlur(image, k)
    return np.clip(result, 0, 255).astype(np.uint8)


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 35.0,
    sigma_space: float = 35.0,
) -> np.ndarray:
    """Denoise while preserving strong edges using a bilateral filter.

    The filter weighs nearby pixels by both spatial distance and color similarity.
    It is generally less likely to smear object boundaries than a mean/Gaussian blur.
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).round().astype(np.uint8)
    d = max(3, int(diameter))
    if d % 2 == 0:
        d += 1
    result = cv2.bilateralFilter(
        img, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space),
        borderType=cv2.BORDER_REFLECT,
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def non_local_means_filter(
    image: np.ndarray,
    noise_strength: float = 0.04,
) -> np.ndarray:
    """Apply OpenCV non-local means denoising with an adaptive strength.

    ``noise_strength`` uses the app's normalized Gaussian-noise scale (0..0.25).
    Non-local means compares similar image patches and often retains fine structure
    better than applying a large median or Gaussian kernel to natural photographs.
    The RGB/BGR conversion is explicit because OpenCV's colored variant expects BGR.
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).round().astype(np.uint8)
    noise_sigma = float(np.clip(noise_strength, 0.0, 0.25)) * 255.0
    h = float(np.clip(2.5 + 0.36 * noise_sigma, 3.0, 12.0))

    if img.ndim == 2:
        result = cv2.fastNlMeansDenoising(
            img, None, h=h, templateWindowSize=7, searchWindowSize=21
        )
    elif img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            bgr, None, h=h, hColor=max(3.0, h * 0.8),
            templateWindowSize=7, searchWindowSize=21,
        )
        result = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Expected a grayscale or RGB uint8 image")
    return np.clip(result, 0, 255).astype(np.uint8)
