"""CityScape segmentation dataset + dataloader (80/20 split, seed 42).

Images live in  data/CameraRGB/*.png   (RGBA, 480x640)
Masks  live in  data/CameraMask/*.png  (RGBA, 480x640, class index stored in R channel)
There are 23 classes in total (values 0..22).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 23
IMG_SIZE = (240, 320)


def list_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    data_dir = Path(data_dir)
    rgb_dir = data_dir / "CameraRGB"
    mask_dir = data_dir / "CameraMask"
    files = sorted(p.name for p in rgb_dir.glob("*.png"))
    pairs = []
    for fname in files:
        r = rgb_dir / fname
        m = mask_dir / fname
        if r.exists() and m.exists():
            pairs.append((r, m))
    return pairs


def split_pairs(
    pairs: Sequence[tuple[Path, Path]],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_test = int(round(len(pairs) * test_size))
    test_idx = set(idx[:n_test])
    train, test = [], []
    for i, p in enumerate(pairs):
        (test if i in test_idx else train).append(p)
    return train, test


class CityscapeSegDataset(Dataset):
    """Reads (image, mask) pairs and returns tensors sized to IMG_SIZE."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        img_size: tuple[int, int] = IMG_SIZE,
        num_classes: int = NUM_CLASSES,
        augment: bool = False,
    ):
        self.pairs = list(pairs)
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rgb(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize(
            (self.img_size[1], self.img_size[0]), Image.BILINEAR
        )
        return np.asarray(img, dtype=np.uint8)

    def _load_mask(self, path: Path) -> np.ndarray:
        m = np.asarray(Image.open(path))
        if m.ndim == 3:
            m = m[..., 0]  # class id is stored in the R channel
        m = np.clip(m, 0, self.num_classes - 1).astype(np.int64)
        m_t = torch.from_numpy(m)[None, None].float()
        m_t = F.interpolate(m_t, size=self.img_size, mode="nearest")
        return m_t[0, 0].long().numpy()

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img = self._load_rgb(img_path)
        mask = self._load_mask(mask_path)

        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


def make_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list, list]:
    pairs = list_pairs(data_dir)
    train_pairs, test_pairs = split_pairs(pairs, test_size=0.2, seed=seed)
    train_ds = CityscapeSegDataset(train_pairs, augment=True)
    test_ds = CityscapeSegDataset(test_pairs, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_pairs, test_pairs


if __name__ == "__main__":
    tr, te, trp, tep = make_loaders()
    print(f"train={len(trp)} test={len(tep)}")
    xb, yb = next(iter(tr))
    print("batch:", xb.shape, xb.dtype, yb.shape, yb.dtype, "mask min/max", int(yb.min()), int(yb.max()))
