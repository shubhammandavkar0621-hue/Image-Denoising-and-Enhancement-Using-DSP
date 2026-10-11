"""Utilities for locating and preparing images from a Kaggle dataset."""
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb_image(source, max_side: int = 1024) -> np.ndarray:
    """Load an image file or file-like object as RGB uint8; downscale very large images."""
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def find_images(root: str | Path) -> list[Path]:
    """Recursively list supported image files beneath root, sorted by path."""
    root_path = Path(root).expanduser()
    if not root_path.exists() or not root_path.is_dir():
        return []
    return sorted(
        (p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )


def describe_dataset_layout(root: str | Path) -> str:
    """Return a friendly message about a dataset folder."""
    images = find_images(root)
    if not images:
        return f"No supported images found under `{root}`. Check the folder path and unzip the Kaggle dataset."
    return f"Found {len(images):,} image(s) under `{root}`."
