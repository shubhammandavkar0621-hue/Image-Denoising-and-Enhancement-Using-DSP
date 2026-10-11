import numpy as np

from modules.recommendations import analyze_reference_image, recommend_configuration


def test_recommendation_returns_supported_controls_for_rgb_image():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :48] = 120
    image[20:75, 25:70] = [220, 90, 45]
    result = recommend_configuration(image, "Gaussian")
    assert result["filter_name"] in {
        "Bilateral (edge-preserving)",
        "Non-local Means (detail-preserving)",
        "Gaussian (spatial)",
    }
    assert 0 <= result["noise_strength"] <= 0.25
    assert result["kernel_size"] in [3, 5, 7, 9]
    assert result["enhancement"] in ["None", "CLAHE"]


def test_salt_and_pepper_recommends_median_filter():
    image = np.tile(np.arange(96, dtype=np.uint8), (96, 1))
    rgb_image = np.stack([image, image, image], axis=-1)
    result = recommend_configuration(rgb_image, "Salt-and-pepper")
    assert result["filter_name"] == "Median (spatial)"
    assert result["kernel_size"] == 3


def test_image_analysis_returns_visual_profile():
    image = np.full((80, 80, 3), 140, dtype=np.uint8)
    stats = analyze_reference_image(image)
    assert stats["brightness_label"] == "Balanced brightness"
    assert "contrast_label" in stats
    assert "detail_label" in stats
    assert 0.0 <= stats["edge_density"] <= 1.0
