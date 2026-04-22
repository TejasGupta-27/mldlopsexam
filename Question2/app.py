"""Two-page Streamlit app for the CityScape UNet segmentation model.

Page 1 - Training & Test Metrics:
    - Training loss / mIoU / mDice curves
    - Final mIoU & mDice on the held-out test set

Page 2 - Predictions:
    - User picks up to 4 images from the test split
    - App shows the original RGB, the ground-truth mask and the predicted mask
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from dataset import IMG_SIZE, NUM_CLASSES, list_pairs, split_pairs
from model import UNet

RUN_DIR = Path("runs")
DATA_DIR = Path("data")

st.set_page_config(page_title="CityScape Segmentation", layout="wide")


# ---------- Cached loaders ----------
@st.cache_resource(show_spinner="Loading UNet model...")
def load_model(checkpoint: str = "runs/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=32).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device, ckpt


@st.cache_data
def load_history():
    with (RUN_DIR / "history.json").open() as f:
        return json.load(f)


@st.cache_data
def load_final_metrics():
    with (RUN_DIR / "final_metrics.json").open() as f:
        return json.load(f)


@st.cache_data
def load_test_pairs():
    pairs = list_pairs(DATA_DIR)
    _, test = split_pairs(pairs, test_size=0.2, seed=42)
    return [(str(a), str(b)) for a, b in test]


# ---------- Helpers ----------
def make_palette(num_classes: int = NUM_CLASSES) -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # class 0 = black
    return pal


PALETTE = make_palette()


def colourize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, NUM_CLASSES - 1)]


def prepare_input(rgb: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(rgb).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def predict_mask(model: UNet, device: torch.device, rgb: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = prepare_input(rgb).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def resize_mask(mask: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8))
    im = im.resize((target[1], target[0]), Image.NEAREST)
    return np.asarray(im)


# ---------- Pages ----------
def page_metrics() -> None:
    st.title("1) Training & Test Metrics")
    st.caption("UNet (base=32, 23 classes) trained on CityScape RGB/mask pairs — 80/20 split, seed=42.")

    try:
        history = load_history()
        final = load_final_metrics()
    except FileNotFoundError:
        st.error("Training artefacts not found. Run `python train.py` first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Epochs trained", final.get("epochs", len(history["train_loss"])))
    c2.metric("Test mIoU",  f"{final['final_test_miou']:.4f}")
    c3.metric("Test mDice", f"{final['final_test_mdice']:.4f}")

    col_a, col_b = st.columns(2)
    col_a.metric("Best Test mIoU",                     f"{final['best_test_miou']:.4f}")
    col_b.metric("Test mDice @ best mIoU epoch",       f"{final['best_test_mdice_at_best_miou']:.4f}")

    st.subheader("Plots (saved during training)")
    img_cols = st.columns(3)
    plot_files = [
        ("Training Loss", RUN_DIR / "loss_curve.png"),
        ("Mean IoU",      RUN_DIR / "miou_curve.png"),
        ("Mean Dice",     RUN_DIR / "mdice_curve.png"),
    ]
    for col, (title, path) in zip(img_cols, plot_files):
        with col:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Missing {path}")

    with st.expander("Per-epoch history"):
        st.json(history)


def page_predict() -> None:
    st.title("2) Model Predictions on Test Images")
    st.caption("Upload up to 4 images from the test set. For each we show the original image, ground-truth mask and predicted mask.")

    try:
        model, device, _ = load_model()
    except FileNotFoundError:
        st.error("`runs/best_model.pt` not found — train the model first.")
        return

    test_pairs = load_test_pairs()
    test_name_to_mask = {Path(a).name: b for a, b in test_pairs}
    test_names = sorted(test_name_to_mask.keys())

    mode = st.radio(
        "Source",
        ["Pick from test set", "Upload files"],
        horizontal=True,
    )

    items: list[tuple[str, np.ndarray, np.ndarray | None]] = []  # (name, rgb, gt_mask or None)

    if mode == "Pick from test set":
        picks = st.multiselect(
            f"Choose up to 4 filenames (test set has {len(test_names)} images)",
            options=test_names,
            default=test_names[:4],
            max_selections=4,
        )
        for name in picks:
            rgb = np.asarray(Image.open(name_to_rgb_path(name, test_pairs)).convert("RGB"))
            gt = np.asarray(Image.open(test_name_to_mask[name]))
            if gt.ndim == 3:
                gt = gt[..., 0]
            items.append((name, rgb, gt))
    else:
        files = st.file_uploader(
            "Upload up to 4 RGB images (.png / .jpg)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if files:
            for f in files[:4]:
                rgb = np.asarray(Image.open(f).convert("RGB"))
                gt = None
                if f.name in test_name_to_mask:
                    g = np.asarray(Image.open(test_name_to_mask[f.name]))
                    gt = g[..., 0] if g.ndim == 3 else g
                items.append((f.name, rgb, gt))

    if not items:
        st.info("Select or upload images to see predictions.")
        return

    for name, rgb, gt in items:
        st.markdown(f"### {name}")
        pred = predict_mask(model, device, rgb)
        pred_full = resize_mask(pred, rgb.shape[:2])

        cols = st.columns(3)
        cols[0].image(rgb, caption="Input image", use_container_width=True)
        if gt is not None:
            cols[1].image(colourize(gt), caption="Ground-truth mask", use_container_width=True)
        else:
            cols[1].info("No ground-truth mask available for this upload.")
        cols[2].image(colourize(pred_full), caption="Predicted mask", use_container_width=True)


def name_to_rgb_path(name: str, test_pairs: list[tuple[str, str]]) -> str:
    for rgb, _ in test_pairs:
        if Path(rgb).name == name:
            return rgb
    return name


# ---------- Sidebar router ----------
PAGES = {
    "Page 1 - Training & Test Metrics": page_metrics,
    "Page 2 - Predictions": page_predict,
}

choice = st.sidebar.radio("Pages", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** UNet (23-class)")
st.sidebar.markdown("**Dataset:** CityScape RGB / Mask")
st.sidebar.markdown("**Split:** 80/20, seed=42")

PAGES[choice]()
