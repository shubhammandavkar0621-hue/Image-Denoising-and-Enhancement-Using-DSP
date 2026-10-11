"""Image DSP Studio — polished interactive DTSP demonstration app."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.data_utils import find_images, load_rgb_image
from modules.noise import add_noise
from modules.spatial_filters import (
    mean_filter, gaussian_filter, median_filter, bilateral_filter, non_local_means_filter,
)
from modules.frequency_filters import frequency_filter, fft_magnitude
from modules.enhancement import (
    contrast_stretch,
    histogram_equalization,
    clahe_enhancement,
    unsharp_mask,
)
from modules.metrics import calculate_metrics
from modules.recommendations import recommend_configuration


# Public Kaggle data is downloaded only when a visitor chooses that source.
# Streamlit caches the dataset root for the lifetime of the running app instance.
KAGGLE_DATASET_HANDLE = "puneet6060/intel-image-classification"


@st.cache_resource(show_spinner=False)
def get_public_kaggle_dataset_root() -> str:
    """Download the public Intel Image Classification dataset on first use."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - handled by deployment requirements
        raise RuntimeError(
            "The kagglehub package is missing. Install dependencies from requirements.txt."
        ) from exc
    return str(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))


@st.cache_data(show_spinner=False)
def get_cached_image_paths(dataset_root: str) -> tuple[str, ...]:
    """Cache recursive image discovery so widget reruns do not rescan every file."""
    return tuple(str(path) for path in find_images(dataset_root))


# -----------------------------------------------------------------------------
# Page setup and design system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Image DSP Studio",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Image DSP Studio — an educational image denoising and enhancement workbench for DTSP.",
    },
)

# Set stable defaults before widget creation. This also lets the recommendation button
# change widget state cleanly on the next rerun without duplicate-default warnings.
_DEFAULT_WIDGET_STATE = {
    "source": "Upload an image",
    "dataset_root": "data/raw",
    "noise_type": "Gaussian",
    "noise_strength": 0.06,
    "noise_seed": 42,
    "filter_name": "Bilateral (edge-preserving)",
    "kernel_size": 3,
    "bilateral_diameter": 7,
    "cutoff": 0.12,
    "filter_order": 2,
    "enhancement": "None",
    "sharpness": 1.0,
}
for _state_key, _default_value in _DEFAULT_WIDGET_STATE.items():
    st.session_state.setdefault(_state_key, _default_value)

# Apply a recommendation on the next rerun, before Streamlit creates the widgets.
# This avoids modifying widget state after those widgets have already been instantiated.
_pending_recommendation = st.session_state.pop("_pending_image_recommendation", None)
if _pending_recommendation:
    for _state_key, _state_value in _pending_recommendation.items():
        st.session_state[_state_key] = _state_value

CUSTOM_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  color-scheme: light;
  --dsp-bg: #f4f7fb;
  --dsp-surface: #ffffff;
  --dsp-ink: #172033;
  --dsp-body: #29364b;
  --dsp-muted: #53647d;
  --dsp-border: #d8e1ee;
  --dsp-primary: #1d4ed8;
  --dsp-primary-hover: #1e40af;
  --dsp-teal: #0f766e;
  --dsp-soft-blue: #eaf1ff;
  --dsp-shadow: 0 8px 24px rgba(20, 35, 65, .07);
}
html { color-scheme: light; scroll-behavior: smooth; }
body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
  background: var(--dsp-bg) !important; color: var(--dsp-ink) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: rgba(244,247,251,.96) !important; }
[data-testid="stToolbar"] { right: 1rem; }
.main .block-container { max-width: 1480px; padding: 2rem 2.35rem 3rem; }

/* Set readable foreground colors explicitly. This also prevents a user's dark-mode
   browser/Streamlit preference from making text disappear on the light surfaces. */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] blockquote,
[data-testid="stText"], .stMarkdown, .stText, .stCaption, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
[data-testid="stSlider"] label, [data-testid="stSelectSlider"] label,
[data-testid="stNumberInput"] label, [data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label, [data-testid="stFileUploader"] label,
[data-testid="stDateInput"] label, [data-testid="stMultiSelect"] label {
  color: var(--dsp-body) !important;
}
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b {
  color: var(--dsp-ink) !important; font-weight: 750 !important;
}
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--dsp-muted) !important; }
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
  font-family: 'Manrope', 'DM Sans', sans-serif !important;
  color: var(--dsp-ink) !important; letter-spacing: -.02em;
}
h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 750 !important; }
[data-testid="stMarkdownContainer"] a, a { color: #1d4ed8 !important; text-underline-offset: 3px; }
a:hover { color: #1e40af !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: #edf2f9 !important; border-right: 1px solid var(--dsp-border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
[data-testid="stSidebar"] * { color: var(--dsp-body); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong, [data-testid="stSidebar"] b {
  color: var(--dsp-ink) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #43536b !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #24334a !important; }

.dsp-hero {
  position: relative; overflow: hidden; margin: .15rem 0 1.5rem; padding: 2rem 2.1rem;
  border: 1px solid #d6e1f1; border-radius: 22px;
  background: linear-gradient(118deg, #ffffff 0%, #f0f5ff 58%, #e8f8f7 100%);
  box-shadow: var(--dsp-shadow);
}
.dsp-hero:after { content: ''; position: absolute; width: 240px; height: 240px; right: -70px; top: -115px; border-radius: 50%; border: 34px solid rgba(29,78,216,.08); pointer-events: none; }
.dsp-eyebrow { display: inline-flex; align-items: center; gap: 7px; color: #1e40af !important; background: #e7efff; border: 1px solid #c8d8ff; padding: 5px 10px; border-radius: 999px; font-size: 11px; line-height: 1.3; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
.dsp-hero h1 { color: #14233b !important; font-size: clamp(1.9rem, 3.3vw, 2.8rem); line-height: 1.12; margin: 15px 0 10px; max-width: 880px; }
.dsp-hero p { color: #43536b !important; font-size: 1rem; max-width: 850px; margin: 0; }
.dsp-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.dsp-chip { padding: 6px 10px; border-radius: 9px; background: #ffffff; border: 1px solid #d6e1f0; color: #263750 !important; font-size: 12px; font-weight: 700; }
.dsp-section-heading { display: flex; align-items: center; gap: 10px; margin: .35rem 0 .8rem; font-family: 'Manrope', sans-serif; font-size: 1.14rem; font-weight: 800; color: #172c4a !important; }
.dsp-section-index { display: inline-flex; flex-shrink: 0; width: 28px; height: 28px; justify-content: center; align-items: center; border-radius: 9px; color: #ffffff !important; background: #1d4ed8; font-size: 12px; font-weight: 800; }
.dsp-muted { color: #53647d !important; font-size: .88rem; }
.dsp-status { display: inline-flex; align-items: center; gap: 7px; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 750; border: 1px solid #9ddcc7; color: #075e48 !important; background: #e9fbf3; }
.dsp-status-dot { height: 7px; width: 7px; border-radius: 50%; background: #059669; }
.dsp-note { padding: 13px 15px; border-radius: 12px; border: 1px solid #bdd5fb; background: #edf5ff; color: #183d70 !important; font-size: 13px; line-height: 1.6; }
.dsp-note strong { color: #102e59 !important; }
.dsp-good { padding: 13px 15px; border-radius: 12px; border: 1px solid #a9e7be; background: #edfcf1; color: #14532d !important; font-size: 13px; line-height: 1.6; }
.dsp-warn { padding: 13px 15px; border-radius: 12px; border: 1px solid #f5d27a; background: #fff8e5; color: #713f12 !important; font-size: 13px; line-height: 1.6; }
.dsp-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--dsp-border); color: #53647d !important; font-size: 12px; line-height: 1.7; }

/* Inputs and select controls */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  min-height: 42px !important; border-radius: 11px !important;
  border: 1px solid #c7d3e4 !important; background: #ffffff !important;
  color: #172033 !important; -webkit-text-fill-color: #172033 !important;
  font-weight: 750 !important; opacity: 1 !important;
  transition: background-color .16s ease, border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}
/* Streamlit versions render button labels as nested paragraphs/spans. Style the
   descendants too, so global Markdown/sidebar rules cannot make labels disappear. */
.stButton > button *, .stDownloadButton > button *,
[data-testid="stFormSubmitButton"] > button *,
[data-testid="stBaseButton-secondary"] * {
  color: #172033 !important; -webkit-text-fill-color: #172033 !important;
  opacity: 1 !important; font-weight: 750 !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stBaseButton-primary"], button[kind="primary"] {
  background: #1d4ed8 !important; color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important; border-color: #1d4ed8 !important;
}
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] *,
[data-testid="stBaseButton-primary"] *, button[kind="primary"] * {
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
  transform: translateY(-1px); border-color: #6489df !important;
  box-shadow: 0 5px 14px rgba(29,78,216,.16);
  background: #eaf1ff !important; color: #173d91 !important;
  -webkit-text-fill-color: #173d91 !important;
}
.stButton > button:hover *, .stDownloadButton > button:hover *,
[data-testid="stBaseButton-secondary"]:hover * {
  color: #173d91 !important; -webkit-text-fill-color: #173d91 !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
  background: #1e40af !important; border-color: #1e40af !important;
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover *, .stDownloadButton > button[kind="primary"]:hover *,
[data-testid="stBaseButton-primary"]:hover *, button[kind="primary"]:hover * {
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
}
.stButton > button:disabled, .stDownloadButton > button:disabled,
[data-testid="stBaseButton-primary"]:disabled, [data-testid="stBaseButton-secondary"]:disabled {
  background: #e8edf5 !important; border-color: #d4dce8 !important;
  color: #59677c !important; -webkit-text-fill-color: #59677c !important; opacity: 1 !important;
}
.stButton > button:disabled *, .stDownloadButton > button:disabled *,
[data-testid="stBaseButton-primary"]:disabled *, [data-testid="stBaseButton-secondary"]:disabled * {
  color: #59677c !important; -webkit-text-fill-color: #59677c !important; opacity: 1 !important;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, button:focus-visible,
input:focus-visible, textarea:focus-visible, [role="combobox"]:focus-visible,
[data-baseweb="slider"]:focus-within {
  outline: 3px solid #2563eb !important; outline-offset: 2px !important; box-shadow: 0 0 0 2px #ffffff !important;
}
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
  color: #172033 !important; background: #ffffff !important; caret-color: #1d4ed8 !important;
}
[data-baseweb="select"] > div {
  background: #ffffff !important; border-color: #bdcbe0 !important; border-radius: 10px !important;
}
[data-baseweb="select"] *, [data-baseweb="popover"] *, [role="listbox"] *, [role="option"] * {
  color: #172033 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background: #ffffff !important; }
[role="option"]:hover { background: #eaf1ff !important; }
[data-testid="stFileUploader"] section {
  border-radius: 10px; border: 1px dashed #9eafc7 !important; background: #ffffff !important;
  transition: border-color .16s ease, background .16s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: #1d4ed8 !important; background: #f3f7ff !important; }
[data-testid="stFileUploader"] section * { color: #263750 !important; }
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: #25364e !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #53647d !important; }
[data-testid="stMetric"] { background: #ffffff !important; border: 1px solid var(--dsp-border); border-radius: 14px; padding: 15px 16px; box-shadow: 0 4px 14px rgba(20,35,65,.045); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #52637b !important; font-size: 12px; font-weight: 750; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #172a46 !important; font-family: 'Manrope', sans-serif; font-weight: 800; }
[data-testid="stMetricDelta"] { font-size: 12px; }
[data-testid="stDataFrame"] { background: #ffffff !important; border: 1px solid var(--dsp-border); border-radius: 12px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 12px; }
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid var(--dsp-border); border-radius: 13px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary span { color: #172033 !important; }
[data-testid="stProgressBar"] > div { background: #dce6f5 !important; }
[data-testid="stProgressBar"] > div > div { background: #1d4ed8 !important; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stAlert"] p { color: #24334a !important; }
[data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #d5dfec !important; }
[data-baseweb="tab"] { color: #455670 !important; font-weight: 700 !important; background: transparent !important; }
[data-baseweb="tab"] p { color: inherit !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #1d4ed8 !important; border-bottom-color: #1d4ed8 !important; }
[data-baseweb="tab-highlight"] { background: #1d4ed8 !important; }
hr { border-color: #dbe3ef !important; }

@media (max-width: 900px) {
  .main .block-container { padding: 1.1rem 1rem 2rem; }
  .dsp-hero { padding: 1.35rem 1.2rem; border-radius: 17px; }
  .dsp-hero h1 { font-size: clamp(1.8rem, 7vw, 2.45rem); }
}
@media (max-width: 640px) {
  .main .block-container { padding: .8rem .75rem 1.5rem; }
  .dsp-hero { margin-bottom: 1rem; }
  .dsp-chip-row { gap: 6px; }
  .dsp-chip { font-size: 11px; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .75rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_heading(number: str, title: str, detail: str | None = None) -> None:
    detail_html = f'<span style="font-size:12px;color:#53647d;font-weight:600;margin-left:auto">{detail}</span>' if detail else ""
    st.markdown(
        f'<div class="dsp-section-heading"><span class="dsp-section-index">{number}</span><span>{title}</span>{detail_html}</div>',
        unsafe_allow_html=True,
    )


def format_score(value: float, digits: int = 2, suffix: str = "") -> str:
    if math.isinf(float(value)):
        return f"∞{suffix}"
    if math.isnan(float(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def make_png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Sidebar — all experiment controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:5px 2px 14px">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="width:38px;height:38px;border-radius:12px;background:#4f46e5;color:white;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:800;box-shadow:0 5px 12px #4f46e52a">∿</div>
            <div><div style="font-family:Manrope,sans-serif;font-weight:800;font-size:17px;color:#15243a">Image DSP Studio</div><div style="font-size:11px;color:#64748b;letter-spacing:.06em">SIGNAL PROCESSING LAB</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 01 · Image source")
    # Only offer the local-folder mode when this runtime actually has images in
    # data/raw. On Streamlit Community Cloud the full dataset is intentionally
    # excluded from Git, so showing this option with an empty folder is misleading.
    local_default_root = Path("data/raw")
    local_dataset_available = bool(get_cached_image_paths(str(local_default_root)))
    source_options = ["Kaggle public dataset"]
    if local_dataset_available:
        source_options.append("Kaggle dataset folder")
    source_options.append("Upload an image")
    if st.session_state.get("source") not in source_options:
        st.session_state["source"] = "Upload an image"

    source = st.radio(
        "Choose where to load the reference image from",
        source_options,
        label_visibility="collapsed",
        key="source",
    )
    if not local_dataset_available:
        st.caption(
            "The local dataset-folder option appears automatically when images are present in "
            "`data/raw`. On this hosted app, use the public Kaggle dataset or upload an image."
        )
    selected_image: Path | None = None
    image_bytes = None
    image_label = ""
    all_paths: list[Path] = []
    dataset_root: str | None = None
    dataset_load_error: str | None = None

    if source == "Upload an image":
        uploaded = st.file_uploader(
            "Upload a reference image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            help="Choose a clean reference image for reproducible denoising experiments.",
            key="uploaded_reference",
        )
        if uploaded is not None:
            image_bytes = uploaded
            image_label = uploaded.name

    elif source == "Kaggle public dataset":
        st.caption(
            "Uses the public Intel Image Classification dataset on Kaggle. "
            "The first download can take a few minutes; it is cached while this app instance stays running."
        )
        if st.button(
            "Load public Kaggle dataset",
            type="primary",
            use_container_width=True,
            key="load_public_kaggle_dataset_button",
        ):
            st.session_state["_public_kaggle_load_requested"] = True

        if st.session_state.get("_public_kaggle_load_requested", False):
            try:
                with st.spinner("Downloading the public Kaggle dataset on first use… this may take a few minutes."):
                    dataset_root = get_public_kaggle_dataset_root()
                st.markdown(
                    '<div class="dsp-status"><span class="dsp-status-dot"></span>Public Kaggle dataset is available</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                dataset_load_error = str(exc)
                st.error("The public Kaggle dataset could not be downloaded.")
                st.caption(
                    "Check the deployment logs and internet access. You can still use Upload an image. "
                    f"Details: {dataset_load_error}"
                )
        else:
            st.info("Click **Load public Kaggle dataset** to make the sample images available in this session.")

    else:
        dataset_root = st.text_input(
            "Dataset folder path",
            help="Use a path relative to the project directory, e.g. data/raw.",
            key="dataset_root",
        )

    # Both the public Kaggle source and a local dataset folder share the same browser.
    if source != "Upload an image" and dataset_root:
        with st.spinner("Scanning dataset folders…"):
            all_paths = [Path(path) for path in get_cached_image_paths(str(dataset_root))]

        if all_paths:
            known_splits = [
                name for name in ("seg_train", "seg_test", "seg_pred")
                if any(name in path.parts for path in all_paths)
            ]
            if known_splits:
                split_options = ["All splits"] + known_splits
                if st.session_state.get("dataset_split_selection") not in split_options:
                    st.session_state["dataset_split_selection"] = "All splits"
                selected_split = st.selectbox(
                    "Dataset split",
                    split_options,
                    help="Choose training, testing, prediction, or all images from the dataset.",
                    key="dataset_split_selection",
                )
                if selected_split != "All splits":
                    all_paths = [path for path in all_paths if selected_split in path.parts]

            if all_paths:
                class_names = ["All categories"] + sorted({path.parent.name for path in all_paths})
                if st.session_state.get("dataset_category") not in class_names:
                    st.session_state["dataset_category"] = "All categories"
                category = st.selectbox("Image category", class_names, key="dataset_category")
                candidates = all_paths if category == "All categories" else [path for path in all_paths if path.parent.name == category]
                max_options = 600
                candidates_for_select = candidates[:max_options]
                if len(candidates) > max_options:
                    st.caption(
                        f"Showing {max_options} of {len(candidates):,} matching images. "
                        "Use image upload for a specific file outside this list."
                    )
                image_options = [str(path) for path in candidates_for_select]
                if st.session_state.get("selected_reference_image") not in image_options:
                    st.session_state["selected_reference_image"] = image_options[0]
                selected_text = st.selectbox(
                    "Reference image",
                    image_options,
                    format_func=lambda value: Path(value).name + (f"  ·  {Path(value).parent.name}" if Path(value).parent.name else ""),
                    key="selected_reference_image",
                )
                selected_image = Path(selected_text)
                image_label = selected_image.name
                st.markdown(
                    f'<div class="dsp-status"><span class="dsp-status-dot"></span>{len(all_paths):,} images available in current selection</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No images were found in the selected dataset split.")
        elif source == "Kaggle dataset folder":
            st.markdown(
                '<div class="dsp-warn"><strong>No images detected.</strong><br>Check that the Kaggle folders are inside <code>data/raw</code>, or update the path above.</div>',
                unsafe_allow_html=True,
            )
        elif source == "Kaggle public dataset" and not dataset_load_error:
            st.info("Load the public Kaggle dataset to browse its images.")

    st.divider()
    st.markdown("#### 02 · Noise model")
    noise_type = st.selectbox("Synthetic noise", ["Gaussian", "Salt-and-pepper", "Speckle"], key="noise_type")
    # The slider has different maxima depending on noise model. Keep its saved value
    # within the active range when users switch between Gaussian/speckle and impulse noise.
    _noise_max = 0.20 if noise_type == "Salt-and-pepper" else 0.25
    st.session_state["noise_strength"] = float(
        np.clip(float(st.session_state.get("noise_strength", 0.06)), 0.0, _noise_max)
    )
    if noise_type == "Salt-and-pepper":
        noise_strength = st.slider("Noise density", 0.0, 0.20, value=None, step=0.01, help="Fraction of pixels replaced by black or white pixels.", key="noise_strength")
    else:
        noise_strength = st.slider("Noise strength", 0.0, 0.25, value=None, step=0.01, help="Standard deviation relative to the normalized 0–1 intensity range.", key="noise_strength")
    noise_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=None, step=1, help="Keep the same seed to reproduce an experiment.", key="noise_seed")

    st.divider()
    st.markdown("#### 03 · Denoising")
    filter_options = [
        "Bilateral (edge-preserving)",
        "Non-local Means (detail-preserving)",
        "Median (spatial)",
        "Mean (spatial)",
        "Gaussian (spatial)",
        "Ideal low-pass (FFT)",
        "Butterworth low-pass (FFT)",
        "Gaussian low-pass (FFT)",
    ]
    filter_name = st.selectbox(
        "Filter algorithm", filter_options,
        help="For natural photos with Gaussian noise, start with Bilateral or Non-local Means. Median filtering is most useful for salt-and-pepper noise.",
        key="filter_name",
    )
    kernel_size = st.select_slider(
        "Spatial kernel size", options=[3, 5, 7, 9], value=None,
        help="Used by Mean, Gaussian and Median spatial filters. Larger kernels smooth more strongly but may erase detail.",
        key="kernel_size",
    )
    if filter_name == "Bilateral (edge-preserving)":
        bilateral_diameter = st.select_slider(
            "Bilateral neighborhood diameter", options=[3, 5, 7, 9], value=None,
            help="Larger neighborhoods can remove more noise but may soften fine texture.",
            key="bilateral_diameter",
        )
    else:
        bilateral_diameter = int(st.session_state.get("bilateral_diameter", 7))
    if noise_type == "Gaussian" and filter_name == "Median (spatial)" and kernel_size >= 5:
        st.warning("A median kernel of 5×5 or larger can blur fine textures with Gaussian noise. Try Bilateral (edge-preserving), Non-local Means, or a 3×3 median filter.", icon="⚠️")
    cutoff = st.slider("FFT cutoff frequency", 0.01, 0.50, value=None, step=0.01, help="Normalized radial frequency cutoff for frequency-domain filters.", key="cutoff")
    filter_order = st.slider("Butterworth order", 1, 8, value=None, step=1, help="Higher orders create a sharper transition and may increase ringing.", key="filter_order")

    st.divider()
    st.markdown("#### 04 · Enhancement")
    enhancement = st.selectbox("Enhancement operation", ["None", "Contrast stretching", "Histogram equalization", "CLAHE", "Unsharp masking"], key="enhancement")
    if enhancement == "Unsharp masking":
        sharpness = st.slider("Sharpening strength", 0.1, 2.5, value=None, step=0.1, key="sharpness")
    else:
        sharpness = 1.0

    st.markdown(
        '<div style="padding:10px 0 0;color:#43536b;font-size:11px;line-height:1.5">Tip: keep the same image and noise seed while comparing filters. This makes your DTSP experiment fair and reproducible.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Header and input handling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <section class="dsp-hero" aria-label="Image DSP Studio overview">
      <span class="dsp-eyebrow">DTSP · INTERACTIVE WORKBENCH</span>
      <h1>Image denoising, made measurable.</h1>
      <p>Explore spatial and frequency-domain filters, enhance image quality, and compare results with objective signal-processing metrics — all in one workspace.</p>
      <div class="dsp-chip-row">
        <span class="dsp-chip">2D DFT / FFT</span>
        <span class="dsp-chip">Spatial filters</span>
        <span class="dsp-chip">Frequency-domain filtering</span>
        <span class="dsp-chip">PSNR · SSIM · MSE</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    if source == "Upload an image":
        if image_bytes is None:
            st.markdown(
                '<div class="dsp-note"><strong>Ready when you are.</strong> Upload a clean reference image in the sidebar to start an experiment. JPG, PNG, BMP, TIFF and WEBP are supported.</div>',
                unsafe_allow_html=True,
            )
            st.stop()
        reference = load_rgb_image(image_bytes)
        image_label = getattr(image_bytes, "name", "uploaded image")
        dataset_count = None
    else:
        if selected_image is None:
            if source == "Kaggle public dataset":
                message = (
                    "<strong>Public dataset not loaded yet.</strong> In the sidebar, choose "
                    "<em>Kaggle public dataset</em> and click <em>Load public Kaggle dataset</em>. "
                    "The first download may take a few minutes."
                )
            else:
                message = (
                    "<strong>Dataset not ready.</strong> Choose a valid local dataset folder path "
                    "in the sidebar, usually <code>data/raw</code>."
                )
            st.markdown(f'<div class="dsp-note">{message}</div>', unsafe_allow_html=True)
            st.stop()
        reference = load_rgb_image(selected_image)
        image_label = selected_image.name
        dataset_count = len(all_paths)
except Exception as exc:
    st.error(f"Could not load the selected image. Details: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Image-based recommendation (transparent heuristic; no external AI/API required)
# -----------------------------------------------------------------------------
recommendation = recommend_configuration(reference, noise_type)


# -----------------------------------------------------------------------------
# DSP pipeline
# -----------------------------------------------------------------------------
noisy = add_noise(reference, noise_type, float(noise_strength), int(noise_seed))
filter_start = time.perf_counter()
if filter_name == "Bilateral (edge-preserving)":
    # Edge-preserving default: avoids the strong texture loss seen with large median kernels.
    sigma_color = float(np.clip(18.0 + float(noise_strength) * 255.0 * 1.4, 18.0, 75.0))
    restored = bilateral_filter(noisy, diameter=bilateral_diameter, sigma_color=sigma_color, sigma_space=35.0)
elif filter_name == "Non-local Means (detail-preserving)":
    restored = non_local_means_filter(noisy, float(noise_strength))
elif filter_name == "Median (spatial)":
    restored = median_filter(noisy, kernel_size)
elif filter_name == "Mean (spatial)":
    restored = mean_filter(noisy, kernel_size)
elif filter_name == "Gaussian (spatial)":
    restored = gaussian_filter(noisy, kernel_size)
elif filter_name == "Ideal low-pass (FFT)":
    restored = frequency_filter(noisy, "Ideal", cutoff, filter_order)
elif filter_name == "Butterworth low-pass (FFT)":
    restored = frequency_filter(noisy, "Butterworth", cutoff, filter_order)
else:
    restored = frequency_filter(noisy, "Gaussian", cutoff, filter_order)
filter_elapsed_ms = (time.perf_counter() - filter_start) * 1000.0

enhancement_start = time.perf_counter()
if enhancement == "Contrast stretching":
    output = contrast_stretch(restored)
elif enhancement == "Histogram equalization":
    output = histogram_equalization(restored)
elif enhancement == "CLAHE":
    output = clahe_enhancement(restored)
elif enhancement == "Unsharp masking":
    output = unsharp_mask(restored, strength=sharpness)
else:
    output = restored
enhancement_elapsed_ms = (time.perf_counter() - enhancement_start) * 1000.0

metrics_noisy = calculate_metrics(reference, noisy)
metrics_restored = calculate_metrics(reference, output)
psnr_gain = metrics_restored["PSNR (dB)"] - metrics_noisy["PSNR (dB)"]
ssim_gain = metrics_restored["SSIM"] - metrics_noisy["SSIM"]
mse_delta = metrics_restored["MSE"] - metrics_noisy["MSE"]

# Clear a comparison table when an input/noise parameter changes, so stale results
# are never presented as if they belonged to the current experiment.
comparison_signature = (
    str(selected_image) if selected_image is not None else image_label,
    noise_type,
    float(noise_strength),
    int(noise_seed),
    float(cutoff),
    int(filter_order),
)
if st.session_state.get("comparison_signature") != comparison_signature:
    st.session_state["comparison_signature"] = comparison_signature
    st.session_state.pop("comparison_table", None)


# -----------------------------------------------------------------------------
# Image-based recommendation note
# -----------------------------------------------------------------------------
section_heading("R", "Suggested configuration for this image", "Automatic image-content analysis")
profile_text = (
    f"Detected profile: **{recommendation['detail_label']}**, "
    f"**{recommendation['brightness_label']}**, **{recommendation['contrast_label']}**. "
    f"Mean brightness is {recommendation['brightness'] * 100:.0f}%, "
    f"grayscale contrast (standard deviation) is {recommendation['contrast_std']:.1f}/255, "
    f"and detected edge density is {recommendation['edge_density'] * 100:.1f}%."
)
st.markdown(
    f"""<div class=\"dsp-note\"><strong>Recommended starting point</strong><br>
    <span>{recommendation['reason']}</span><br><br>
    <strong>Suggested settings:</strong> {recommendation['filter_name']} · 
    {recommendation['noise_type']} noise at {recommendation['noise_strength']:.2f} · 
    {recommendation['enhancement']} enhancement.<br>
    <strong>Parameters:</strong> {recommendation['parameter_summary']}<br>
    <span>{recommendation['enhancement_reason']}</span><br><br>
    <strong>Image analysis:</strong> {profile_text.replace('**', '')}<br>
    <span>These are heuristic starting settings based on brightness, contrast and edge/detail measurements. A clean reference cannot reveal the real noise type; the selected noise is synthetic for this experiment. Compare the output and metrics before deciding.</span>
    </div>""",
    unsafe_allow_html=True,
)
rec_button_col, rec_hint_col = st.columns([1, 2], gap="medium")
with rec_button_col:
    if st.button("Apply suggested configuration", type="primary", use_container_width=True, key="apply_image_recommendation"):
        st.session_state["_pending_image_recommendation"] = {
            "noise_strength": recommendation["noise_strength"],
            "filter_name": recommendation["filter_name"],
            "kernel_size": recommendation["kernel_size"],
            "bilateral_diameter": recommendation["bilateral_diameter"],
            "enhancement": recommendation["enhancement"],
            "cutoff": recommendation["cutoff"],
            "filter_order": recommendation["filter_order"],
        }
        st.rerun()
with rec_hint_col:
    st.markdown(
        '<div class="dsp-muted" style="padding:10px 0">Applying updates the sidebar settings. You can still change every setting manually afterwards.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Top-level experiment summary
# -----------------------------------------------------------------------------
section_heading("A", "Current experiment", "Reproducible reference-based test")
meta_cols = st.columns([1.8, 1, 1, 1], gap="medium")
with meta_cols[0]:
    st.markdown(f"**{image_label}**")
    st.caption(f"{reference.shape[1]:,} × {reference.shape[0]:,} px · RGB · 8-bit")
with meta_cols[1]:
    st.markdown("<div class='dsp-muted'>Noise model</div>", unsafe_allow_html=True)
    st.markdown(f"**{noise_type}**")
    st.caption(f"Strength {noise_strength:.2f} · seed {int(noise_seed)}")
with meta_cols[2]:
    st.markdown("<div class='dsp-muted'>Denoising</div>", unsafe_allow_html=True)
    st.markdown(f"**{filter_name}**")
    st.caption(f"Kernel {kernel_size} × {kernel_size}")
with meta_cols[3]:
    st.markdown("<div class='dsp-muted'>Pipeline runtime</div>", unsafe_allow_html=True)
    st.markdown(f"**{filter_elapsed_ms + enhancement_elapsed_ms:.2f} ms**")
    st.caption("Filter + enhancement")


# -----------------------------------------------------------------------------
# Results / analysis / benchmark tabs
# -----------------------------------------------------------------------------
tab_results, tab_analysis, tab_compare = st.tabs(["Results overview", "DSP analysis", "Filter benchmark"])

with tab_results:
    section_heading("01", "Visual comparison", "Same reference · same synthetic noise")
    col_ref, col_noisy, col_output = st.columns(3, gap="medium")
    with col_ref:
        st.markdown("**Reference image**")
        st.image(reference, use_container_width=True)
        st.caption("Clean source used as the evaluation reference.")
    with col_noisy:
        st.markdown("**Noisy input**")
        st.image(noisy, use_container_width=True)
        st.caption("Synthetic corruption applied reproducibly.")
    with col_output:
        st.markdown("**Processed output**")
        st.image(output, use_container_width=True)
        st.caption(f"{filter_name}" + (f" + {enhancement}" if enhancement != "None" else ""))

    section_heading("02", "Quality metrics", "Higher PSNR / SSIM and lower MSE are generally better")
    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric(
        "PSNR · processed",
        format_score(metrics_restored["PSNR (dB)"], 2, " dB"),
        f"{psnr_gain:+.2f} dB vs noisy" if not math.isinf(psnr_gain) else "Perfect match vs reference",
        delta_color="normal",
        help="Peak signal-to-noise ratio relative to the original clean reference image.",
    )
    metric_cols[1].metric(
        "SSIM · processed",
        format_score(metrics_restored["SSIM"], 4),
        f"{ssim_gain:+.4f} vs noisy",
        delta_color="normal",
        help="Structural Similarity Index; values closer to 1 generally indicate more similar structure.",
    )
    metric_cols[2].metric(
        "MSE · processed",
        format_score(metrics_restored["MSE"], 2),
        f"{mse_delta:+.2f} vs noisy",
        delta_color="inverse",
        help="Mean squared pixel error relative to the clean reference image.",
    )
    metric_cols[3].metric(
        "Filter runtime",
        format_score(filter_elapsed_ms, 2, " ms"),
        f"Enhancement {enhancement_elapsed_ms:.2f} ms",
        help="Runtime varies with image dimensions, hardware, and implementation.",
    )

    st.markdown(
        '<div class="dsp-note"><strong>How to interpret this:</strong> these full-reference metrics are valid because this experiment begins with a reference image and adds synthetic noise. A better metric score does not automatically mean the image looks better in every context, so inspect visual detail and edge preservation too.</div>',
        unsafe_allow_html=True,
    )

    download_cols = st.columns([1, 1, 2], gap="medium")
    with download_cols[0]:
        st.download_button(
            "↓ Download processed PNG",
            data=make_png_bytes(output),
            file_name="denoised_enhanced.png",
            mime="image/png",
            use_container_width=True,
        )
    with download_cols[1]:
        st.download_button(
            "↓ Download noisy PNG",
            data=make_png_bytes(noisy),
            file_name="synthetic_noisy_input.png",
            mime="image/png",
            use_container_width=True,
        )

with tab_analysis:
    section_heading("03", "Frequency-domain inspection", "Centered 2-D FFT magnitude")
    st.markdown(
        "The logarithmic magnitude spectrum shows how spatial-frequency energy is distributed. Low spatial frequencies appear near the center after shifting; higher frequencies appear farther away. Noise can add energy across a broad frequency range.",
    )
    spectrum_cols = st.columns(2, gap="large")
    with spectrum_cols[0]:
        st.markdown("**Reference spectrum**")
        st.image(fft_magnitude(reference), clamp=True, use_container_width=True)
    with spectrum_cols[1]:
        st.markdown("**Noisy spectrum**")
        st.image(fft_magnitude(noisy), clamp=True, use_container_width=True)

    st.divider()
    section_heading("04", "Filter design notes", "Current sidebar settings")
    notes_cols = st.columns(3, gap="medium")
    notes_cols[0].markdown("**Spatial-domain filters**")
    notes_cols[0].write("Mean and Gaussian filters smooth neighborhoods. Median filtering is often more robust to isolated salt-and-pepper outliers, but can still remove fine texture.")
    notes_cols[1].markdown("**Frequency-domain filters**")
    notes_cols[1].write(f"The selected FFT cutoff is **{cutoff:.2f}** of the smaller image dimension. The Butterworth order is **{filter_order}**. Lower cutoffs remove more high-frequency detail as well as noise.")
    notes_cols[2].markdown("**Enhancement stage**")
    notes_cols[2].write(f"Current enhancement: **{enhancement}**. Enhancement changes appearance and can alter metric scores; sharpened images may also amplify noise.")

with tab_compare:
    section_heading("05", "Benchmark denoising methods", "Every method uses the same noisy input")
    st.markdown("Run a controlled comparison across spatial and frequency-domain methods. The results stay available while you explore other tabs, and reset when the image or noise configuration changes.")
    run_col, info_col = st.columns([1, 2], gap="medium")
    with run_col:
        run_comparison = st.button("Run 9-method comparison", type="primary", use_container_width=True)
    with info_col:
        st.markdown('<div class="dsp-muted" style="padding:9px 0">Includes a no-filter baseline, edge-preserving methods, spatial filters and FFT filters. Processing time is measured separately for each method.</div>', unsafe_allow_html=True)

    if run_comparison:
        methods = [
            ("No denoising", lambda: noisy),
            ("Non-local means", lambda: non_local_means_filter(noisy, float(noise_strength))),
            ("Bilateral", lambda: bilateral_filter(noisy, diameter=7, sigma_color=float(np.clip(18.0 + float(noise_strength) * 255.0 * 1.4, 18.0, 75.0)), sigma_space=35.0)),
            ("Mean 3×3", lambda: mean_filter(noisy, 3)),
            ("Gaussian 3×3", lambda: gaussian_filter(noisy, 3)),
            ("Median 3×3", lambda: median_filter(noisy, 3)),
            ("Ideal low-pass FFT", lambda: frequency_filter(noisy, "Ideal", cutoff, filter_order)),
            ("Butterworth low-pass FFT", lambda: frequency_filter(noisy, "Butterworth", cutoff, filter_order)),
            ("Gaussian low-pass FFT", lambda: frequency_filter(noisy, "Gaussian", cutoff, filter_order)),
        ]
        result_rows = []
        progress = st.progress(0, text="Preparing benchmark…")
        try:
            for idx, (method_name, method) in enumerate(methods):
                started = time.perf_counter()
                candidate = method()
                elapsed = (time.perf_counter() - started) * 1000.0
                scores = calculate_metrics(reference, candidate)
                result_rows.append({"Method": method_name, **scores, "Time (ms)": elapsed})
                progress.progress((idx + 1) / len(methods), text=f"Evaluated {idx + 1} of {len(methods)} methods")
            comparison = pd.DataFrame(result_rows).sort_values("PSNR (dB)", ascending=False).reset_index(drop=True)
            st.session_state["comparison_table"] = comparison
            progress.empty()
            st.success("Benchmark complete. Results are ready below.")
        except Exception as exc:
            progress.empty()
            st.error(f"The benchmark could not finish: {exc}")

    comparison = st.session_state.get("comparison_table")
    if comparison is not None:
        best_row = comparison.iloc[0]
        best_cols = st.columns(3, gap="medium")
        best_cols[0].metric("Top PSNR method", str(best_row["Method"]))
        best_cols[1].metric("Best PSNR", format_score(best_row["PSNR (dB)"], 2, " dB"))
        best_ssim_row = comparison.sort_values("SSIM", ascending=False).iloc[0]
        best_cols[2].metric("Best SSIM method", str(best_ssim_row["Method"]))

        st.dataframe(
            comparison.style.format({"MSE": "{:.2f}", "PSNR (dB)": "{:.2f}", "SSIM": "{:.4f}", "Time (ms)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        chart_cols = st.columns(2, gap="large")
        with chart_cols[0]:
            st.markdown("**PSNR by method (dB)**")
            st.bar_chart(comparison.set_index("Method")[["PSNR (dB)"]], use_container_width=True)
        with chart_cols[1]:
            st.markdown("**SSIM by method**")
            st.bar_chart(comparison.set_index("Method")[["SSIM"]], use_container_width=True)

        st.download_button(
            "↓ Download benchmark CSV",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="dsp_filter_comparison.csv",
            mime="text/csv",
            use_container_width=False,
        )
    else:
        st.markdown(
            '<div class="dsp-note"><strong>No benchmark run yet.</strong> Select <em>Run 9-method comparison</em> to create a table and charts for your report.</div>',
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="dsp-footer">
      <strong>Image DSP Studio</strong> · DTSP academic project · Synthetic-noise experiments are reproducible using the selected random seed.<br>
      Results vary with image content, noise model, noise strength, filter parameters, and hardware. Inspect outputs visually as well as numerically.
    </div>
    """,
    unsafe_allow_html=True,
)
