"""Synthetic noise generation for reproducible image-denoising experiments."""
import numpy as np


def _as_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind in "ui":
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def add_noise(image: np.ndarray, noise_type: str, strength: float, seed: int = 42) -> np.ndarray:
    """Return a uint8 copy of image with synthetic noise.

    strength is noise standard deviation for Gaussian/speckle (0..0.25), or
    salt-and-pepper corruption probability (0..0.20).
    """
    base = _as_float(image)
    rng = np.random.default_rng(seed)
    kind = noise_type.strip().lower()

    if kind == "gaussian":
        noisy = base + rng.normal(0.0, strength, size=base.shape).astype(np.float32)
    elif kind in {"salt-and-pepper", "salt & pepper", "salt and pepper"}:
        noisy = base.copy()
        if base.ndim == 2:
            random_map = rng.random(base.shape)
        else:
            random_map = rng.random(base.shape[:2])
        pepper = random_map < (strength / 2.0)
        salt = (random_map >= (strength / 2.0)) & (random_map < strength)
        noisy[pepper] = 0.0
        noisy[salt] = 1.0
    elif kind == "speckle":
        noisy = base + base * rng.normal(0.0, strength, size=base.shape).astype(np.float32)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    return np.clip(noisy * 255.0, 0, 255).round().astype(np.uint8)
