# Image DSP Studio — DTSP Project

An interactive Python application for comparing image denoising and enhancement algorithms using spatial-domain and frequency-domain DSP techniques.

## UI refresh

The Streamlit interface uses a neutral slate background, indigo accent, clear typography, responsive columns, consistent rounded controls, hover/focus feedback, high-contrast status messages, keyboard focus indicators, and reduced-motion preferences. The core image-processing modules are kept separate from the interface.

## Features

- Load images from a local Kaggle dataset folder or upload an image.
- Generate Gaussian, salt-and-pepper, and speckle noise reproducibly.
- Apply mean, Gaussian, and median spatial filters.
- Apply ideal, Butterworth, and Gaussian low-pass filtering in the 2D FFT domain.
- Apply contrast stretching, histogram equalization, CLAHE, and unsharp masking.
- Inspect 2D FFT magnitude spectra.
- Show automatic, image-based starter recommendations with an apply button.
- Compare MSE, PSNR, SSIM, and runtime.
- Run a nine-method filter benchmark and export CSV results.
- Download noisy and processed images.

## Requirements

Python 3.10+ is recommended. Python 3.11 is a good choice for Windows.

## Windows / PowerShell setup

Open a terminal in the folder containing `app.py`:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell activation is blocked by policy, skip activation and use:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Kaggle data layout

Place the extracted Intel Image Classification dataset folders under `data/raw/`, for example:

```text
data/raw/
├── seg_pred/
├── seg_test/
└── seg_train/
```

In the sidebar, choose **Kaggle dataset folder** and set the folder path to `data/raw`. Image lookup is recursive, so nested class folders are supported. You can also select **Upload an image**.

## Experiment notes

For reference-based evaluation, the app starts with a source image, adds synthetic noise, then compares processing results against the original source. This makes PSNR, SSIM, and MSE meaningful for the simulated experiment. For naturally noisy images without a clean reference, do not interpret full-reference metrics as ground truth.

## Batch experiment for the report

```powershell
python run_experiments.py --data-dir data/raw --max-images 25 --noise gaussian --strength 0.06
```

The batch script writes CSV files and a comparison chart under `results/`.

## Tests

```powershell
python -m pip install pytest
python -m pytest
```

## Readable light theme

The app includes a high-contrast light color scheme in `.streamlit/config.toml`, plus CSS overrides for the sidebar, labels, select menus, tabs, metrics, captions, and buttons. If you update the theme while Streamlit is already running, stop the server with `Ctrl+C` and restart it with `python -m streamlit run app.py`.



## Choosing a denoising filter

- **Bilateral (edge-preserving)** is the default for natural color photographs because it smooths similar neighboring pixels while retaining stronger edges.
- **Non-local Means** compares similar patches and is another useful choice for Gaussian-like noise.
- **Median** is especially suitable for salt-and-pepper noise. Large kernels (5×5 and above) can erase fine texture, so start with 3×3 when comparing it against other methods.
- Keep the same source image, noise type, noise strength, and random seed when comparing algorithms. Check both visual details and the reference-based PSNR/SSIM results.

## Image-based configuration recommendation

The dashboard now analyses the selected reference image for mean brightness, grayscale contrast, Canny edge density, and Laplacian variance. It shows a visible recommendation note with an explainable starter denoiser, strength, and enhancement, and an **Apply suggested configuration** button. This is a local heuristic recommender, not a trained ML model or external AI service. It cannot infer unknown real-world noise from a clean reference; the noise types in this application are synthetic experimental settings.
