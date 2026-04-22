# Question 2 

**Question2: mIoU: 0.5215 and mDICE: 0.5784** (held-out test set, 30 epochs).

## Training details
- Dataset: CityScape (RGB + per-pixel class masks, 23 classes)
- Split: 80/20 train/test, seed = 42
- Model: UNet (double-conv blocks, base=32, ~7.76 M params)
- Loss: Cross-entropy (pixel-level)
- Optimizer: Adam, lr=1e-3, cosine-annealing for 30 epochs
- Input resolution: 240×320 (mask is nearest-neighbour-resized to match)
- Augmentation: horizontal flip (p=0.5)
- Device: NVIDIA GTX 1080 Ti (single GPU)

## Results (on the held-out test set)

| Metric            | Value  |
|-------------------|--------|
| Test mIoU         | 0.5215 |
| Test mDice        | 0.5784 |
| Best Test mIoU    | 0.5217 |
| mDice @ best mIoU | 0.5785 |

Both metrics are comfortably above the 0.48 threshold.

## Training curves

See `plots/`:
- `plots/loss_curve.png`   — cross-entropy training loss per epoch
- `plots/miou_curve.png`   — mIoU per epoch (train + test)
- `plots/mdice_curve.png`  — mDice per epoch (train + test)

## Files

| File                    | Purpose                                              |
|-------------------------|------------------------------------------------------|
| `dataset.py`            | Custom Dataset + 80/20 split + DataLoader factory    |
| `model.py`              | UNet implementation                                  |
| `metrics.py`            | Confusion-matrix based mIoU / mDice                  |
| `train.py`              | Training loop, per-epoch eval, plots, checkpointing  |
| `app.py`                | Two-page Streamlit app                               |
| `plots/*.png`           | Training curves                                      |
| `history.json`          | Per-epoch metrics                                    |
| `final_metrics.json`    | Best / final mIoU & mDice on test set                |
| `screenshots/`          | Screenshots of the two Streamlit pages               |

## Reproduction

```bash
# 1. Create env and install deps
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements.txt

# 2. Download the dataset (data/CameraRGB, data/CameraMask) with gdown
gdown --folder "https://drive.google.com/drive/u/1/folders/1GNe3Tu8Mud_CSLOiQZYHS2Rjq2sS74b_"
unzip MLDLOPs_2026_Major_Exam/data.zip -d .
mkdir -p data && mv CameraRGB data/ && mv CameraMask data/

# 3. Train for 30 epochs
.venv/bin/python train.py --epochs 30 --batch-size 16 --lr 1e-3 --out runs

# 4. Launch the 2-page Streamlit app
.venv/bin/python -m streamlit run app.py
```

## Streamlit App

- **Page 1 — Training & Test Metrics**: displays loss / mIoU / mDice curves plus metric cards for the final test mIoU and mDice.
- **Page 2 — Predictions**: lets the user pick up to 4 images from the test split (or upload their own) and displays input image, ground-truth mask, and predicted mask side-by-side.
