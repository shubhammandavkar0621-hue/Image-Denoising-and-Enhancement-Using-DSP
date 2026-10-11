import numpy as np
from modules.noise import add_noise
from modules.spatial_filters import mean_filter, gaussian_filter, median_filter, bilateral_filter, non_local_means_filter
from modules.frequency_filters import frequency_filter, fft_magnitude
from modules.enhancement import contrast_stretch, histogram_equalization, clahe_enhancement, unsharp_mask
from modules.metrics import calculate_metrics


def sample_image():
    x = np.zeros((64, 64, 3), dtype=np.uint8)
    x[16:48, 16:48] = [180, 120, 70]
    return x


def test_noise_is_reproducible():
    image = sample_image()
    assert np.array_equal(add_noise(image, "Gaussian", 0.05, seed=7), add_noise(image, "Gaussian", 0.05, seed=7))


def test_filters_keep_shape_and_dtype():
    image = add_noise(sample_image(), "Gaussian", 0.05)
    methods = [mean_filter(image), gaussian_filter(image), median_filter(image),
               bilateral_filter(image), non_local_means_filter(image),
               frequency_filter(image, "Butterworth", 0.12, 2),
               frequency_filter(image, "Gaussian", 0.12, 2), frequency_filter(image, "Ideal", 0.12, 2)]
    for result in methods:
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_enhancement_and_metrics():
    image = sample_image()
    for result in [contrast_stretch(image), histogram_equalization(image), clahe_enhancement(image), unsharp_mask(image)]:
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    scores = calculate_metrics(image, image)
    assert scores["MSE"] == 0.0
    assert scores["SSIM"] > 0.99
    assert fft_magnitude(image).shape == image.shape[:2]
