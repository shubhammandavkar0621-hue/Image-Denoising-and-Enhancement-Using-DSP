"""Batch-evaluate denoising methods on a small sample of images from a dataset folder.

Example:
    python run_experiments.py --data-dir data/raw --max-images 25 --noise gaussian --strength 0.06
"""
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from modules.data_utils import find_images, load_rgb_image
from modules.noise import add_noise
from modules.spatial_filters import mean_filter, gaussian_filter, median_filter, bilateral_filter, non_local_means_filter
from modules.frequency_filters import frequency_filter
from modules.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Root folder with Kaggle image dataset")
    parser.add_argument("--max-images", type=int, default=25, help="Maximum images to evaluate")
    parser.add_argument("--noise", choices=["gaussian", "salt-and-pepper", "speckle"], default="gaussian")
    parser.add_argument("--strength", type=float, default=0.06, help="Gaussian/speckle sigma or salt-and-pepper density")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    image_paths = find_images(args.data_dir)
    if not image_paths:
        raise SystemExit(f"No images found in '{args.data_dir}'. Download and extract the Kaggle dataset first.")
    image_paths = image_paths[:max(1, args.max_images)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("No denoising", lambda image, strength: image),
        ("Non-local means", lambda image, strength: non_local_means_filter(image, strength)),
        ("Bilateral", lambda image, strength: bilateral_filter(image, diameter=7, sigma_color=float(max(18.0, min(75.0, 18.0 + strength * 255.0 * 1.4))), sigma_space=35.0)),
        ("Mean 3x3", lambda image, strength: mean_filter(image, 3)),
        ("Gaussian 3x3", lambda image, strength: gaussian_filter(image, 3)),
        ("Median 3x3", lambda image, strength: median_filter(image, 3)),
        ("Ideal low-pass FFT", lambda image, strength: frequency_filter(image, "Ideal", 0.12, 2)),
        ("Butterworth low-pass FFT", lambda image, strength: frequency_filter(image, "Butterworth", 0.12, 2)),
        ("Gaussian low-pass FFT", lambda image, strength: frequency_filter(image, "Gaussian", 0.12, 2)),
    ]
    rows = []
    for image_index, path in enumerate(image_paths):
        reference = load_rgb_image(path)
        noisy = add_noise(reference, args.noise, args.strength, seed=args.seed + image_index)
        for method_name, method in methods:
            started = time.perf_counter()
            restored = method(noisy, args.strength)
            elapsed = (time.perf_counter() - started) * 1000.0
            scores = calculate_metrics(reference, restored)
            rows.append({"Image": path.name, "Category": path.parent.name, "Noise": args.noise,
                         "Strength": args.strength, "Method": method_name, **scores, "Time (ms)": elapsed})
        print(f"Processed {image_index + 1}/{len(image_paths)}: {path.name}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "batch_filter_results.csv"
    df.to_csv(csv_path, index=False)
    summary = df.groupby("Method", as_index=False).agg({"MSE": "mean", "PSNR (dB)": "mean", "SSIM": "mean", "Time (ms)": "mean"})
    summary_path = output_dir / "batch_filter_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(summary["Method"], summary["PSNR (dB)"])
    axes[0].set_title("Mean PSNR by filter")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].tick_params(axis="x", labelrotation=65)
    axes[1].bar(summary["Method"], summary["SSIM"])
    axes[1].set_title("Mean SSIM by filter")
    axes[1].set_ylabel("SSIM")
    axes[1].tick_params(axis="x", labelrotation=65)
    fig.tight_layout()
    plot_path = output_dir / "filter_metric_comparison.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nMean results:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved detailed results: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    main()
