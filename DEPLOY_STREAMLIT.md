# Deploy Image DSP Studio for free on Streamlit Community Cloud

## What is included
- `app.py` and `modules/`: application and DSP algorithms
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: readable light theme
- `tests/`: optional tests
- Default input source is **Upload an image**, because the full Kaggle dataset is deliberately not committed to GitHub.

## 1. Create a GitHub repository
1. Sign in to GitHub and create a new repository named `image-dsp-studio`.
2. For a public demo, choose **Public**. Do not initialize it with a README, license, or `.gitignore` because these files are already present here.
3. Install Git for Windows if `git --version` isn't recognized.

## 2. Push this folder to GitHub
Open PowerShell in the folder containing this file (the folder with `app.py` and `requirements.txt`) and run:

```powershell
git init
git add .
git status
git commit -m "Initial Image DSP Studio app"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/image-dsp-studio.git
git push -u origin main
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username. Before running `git commit`, check `git status`: `data/raw` must not list the Kaggle images and `.streamlit/config.toml` should be included. If Git asks for your identity, set it with `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`.

## 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose repository `YOUR-GITHUB-USERNAME/image-dsp-studio`, branch `main`, and main file path `app.py`.
4. In Advanced settings, choose a supported Python version compatible with the project (Python 3.11 is a sensible starting point).
5. Click **Deploy** and wait for the build to finish.
6. Copy the generated `*.streamlit.app` URL to share your demo.

Community Cloud installs the packages listed in `requirements.txt`. The app's `.streamlit/config.toml` is included to preserve the light theme.

## Dataset note
The full Kaggle Intel Image Classification dataset is intentionally excluded from the Git repository to keep it small. On the deployed app, choose **Upload an image** and upload a reference image. The local Kaggle folder option remains useful when running locally with `data/raw/seg_train`, `data/raw/seg_test`, and `data/raw/seg_pred` present.

## Update the deployed app later
Edit code in this folder, then run:

```powershell
git add .
git commit -m "Update Image DSP Studio"
git push
```

Community Cloud redeploys when the GitHub repository is updated.
