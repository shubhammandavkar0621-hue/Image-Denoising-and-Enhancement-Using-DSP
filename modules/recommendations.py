"""Transparent, image-content-based starter recommendations for Image DSP Studio.

This module does not claim to identify unknown real-world noise. The app's reference
image is treated as clean and synthetic noise is added for controlled experiments.
Recommendations use basic image statistics (brightness, contrast, edge density and
Laplacian variance) to choose a reasonable starting filter and settings.
"""
from __future__ import annotations

import cv2
import numpy as np

FILTER_OPTIONS = {
    "bilateral": "Bilateral (edge-preserving)",
    "nlm": "Non-local Means (detail-preserving)",
    "median": "Median (spatial)",
    "gaussian": "Gaussian (spatial)",
}


def analyze_reference_image(image: np.ndarray) -> dict[str, float | str]:
    """Measure simple visual characteristics from an RGB/grayscale uint8 image."""
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("The reference image is empty.")

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = np.clip(arr[..., :3], 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 2:
        gray = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Expected an RGB or grayscale image.")

    gray_float = gray.astype(np.float32)
    brightness = float(gray_float.mean() / 255.0)
    contrast = float(gray_float.std())
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_map = cv2.Canny(gray, threshold1=70, threshold2=140)
    edge_density = float(np.count_nonzero(edge_map) / edge_map.size)

    # These thresholds are practical heuristics, not universal perceptual standards.
    if brightness < 0.34:
        brightness_label = "Dark"
    elif brightness > 0.70:
        brightness_label = "Bright"
    else:
        brightness_label = "Balanced brightness"

    if contrast < 32:
        contrast_label = "Low contrast"
    elif contrast > 65:
        contrast_label = "High contrast"
    else:
        contrast_label = "Moderate contrast"

    if edge_density >= 0.105 or laplacian_variance >= 450:
        detail_label = "High detail / many edges"
        detail_level = "high"
    elif edge_density >= 0.045 or laplacian_variance >= 140:
        detail_label = "Moderate detail"
        detail_level = "moderate"
    else:
        detail_label = "Relatively smooth / low detail"
        detail_level = "low"

    return {
        "brightness": brightness,
        "brightness_label": brightness_label,
        "contrast_std": contrast,
        "contrast_label": contrast_label,
        "edge_density": edge_density,
        "laplacian_variance": laplacian_variance,
        "detail_label": detail_label,
        "detail_level": detail_level,
    }


def recommend_configuration(image: np.ndarray, noise_type: str = "Gaussian") -> dict:
    """Suggest explainable starting parameters for the selected synthetic-noise model."""
    features = analyze_reference_image(image)
    noise_key = noise_type.strip().lower()
    detail = features["detail_level"]

    # Filter choice is adapted to the noise model selected by the user, because a
    # clean reference image cannot tell us which type of real-world noise is present.
    if noise_key in {"salt-and-pepper", "salt and pepper", "salt & pepper"}:
        filter_name = FILTER_OPTIONS["median"]
        bilateral_diameter = 7
        reason = (
            "Salt-and-pepper corruption creates isolated black/white outliers; a 3×3 "
            "median filter is a sensible first test and is less aggressive than larger kernels."
        )
        suggested_strength = 0.03 if detail == "high" else 0.05
    elif detail == "high":
        filter_name = FILTER_OPTIONS["bilateral"]
        bilateral_diameter = 5
        reason = (
            "The image contains many edges or fine textures. Bilateral filtering is a useful "
            "starting point because it smooths similar neighboring pixels while limiting "
            "averaging across stronger edges."
        )
        suggested_strength = 0.03
    elif detail == "moderate":
        filter_name = FILTER_OPTIONS["nlm"]
        bilateral_diameter = 7
        reason = (
            "The image has a moderate amount of detail. Non-local Means compares similar "
            "patches and is worth testing when you want denoising without using a large blur kernel."
        )
        suggested_strength = 0.04
    else:
        filter_name = FILTER_OPTIONS["gaussian"]
        bilateral_diameter = 7
        reason = (
            "The image appears relatively smooth with fewer detected edges. A small 3×3 "
            "Gaussian filter is a simple baseline; compare it with Bilateral to check detail loss."
        )
        suggested_strength = 0.05

    # Enhancement recommendation is based on brightness/contrast statistics.
    if features["contrast_label"] == "Low contrast" or features["brightness_label"] == "Dark":
        enhancement = "CLAHE"
        enhancement_reason = (
            "The reference image is dark or has low global contrast, so CLAHE is suggested "
            "as an optional local-contrast enhancement after denoising. Check that it does not "
            "make residual noise more visible."
        )
    else:
        enhancement = "None"
        enhancement_reason = "The brightness/contrast statistics do not strongly indicate that enhancement is needed."

    if filter_name == FILTER_OPTIONS["bilateral"]:
        parameter_summary = (
            f"bilateral diameter {bilateral_diameter}, color sigma about "
            f"{float(np.clip(18.0 + suggested_strength * 255.0 * 1.4, 18.0, 75.0)):.0f}, "
            "spatial sigma 35"
        )
    elif filter_name in {FILTER_OPTIONS["median"], FILTER_OPTIONS["gaussian"]}:
        parameter_summary = "3×3 spatial kernel"
    else:
        parameter_summary = "adaptive denoising strength derived from the selected noise level"

    return {
        **features,
        "noise_type": noise_type,
        "noise_strength": float(suggested_strength),
        "filter_name": filter_name,
        "kernel_size": 3,
        "bilateral_diameter": int(bilateral_diameter),
        "parameter_summary": parameter_summary,
        "cutoff": 0.12,
        "filter_order": 2,
        "enhancement": enhancement,
        "reason": reason,
        "enhancement_reason": enhancement_reason,
    }
