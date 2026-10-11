"""Contrast and sharpness enhancement operations."""
import cv2
import numpy as np


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.asarray(image).copy().astype(np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).round().astype(np.uint8)


def _apply_luminance_operation(image: np.ndarray, operation) -> np.ndarray:
    """Apply an OpenCV operation to luminance to avoid changing colors excessively."""
    img = np.asarray(image).astype(np.uint8)
    if img.ndim == 2:
        return operation(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = operation(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    return _apply_luminance_operation(image, cv2.equalizeHist)


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return _apply_luminance_operation(image, clahe.apply)


def unsharp_mask(image: np.ndarray, strength: float = 1.0, kernel_size: int = 5) -> np.ndarray:
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(image, (k, k), sigmaX=0, borderType=cv2.BORDER_REFLECT)
    sharpened = cv2.addWeighted(np.asarray(image), 1.0 + float(strength), blurred, -float(strength), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
