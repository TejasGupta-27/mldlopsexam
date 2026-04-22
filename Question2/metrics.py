"""Segmentation metrics: mean IoU (Jaccard) and mean Dice.

Uses a confusion matrix over pixels to compute per-class IoU/Dice, then
averages over classes that actually appear in the reference or prediction.
"""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # pred: (N, H, W) int64   target: (N, H, W) int64
        p = pred.detach().cpu().numpy().ravel()
        t = target.detach().cpu().numpy().ravel()
        mask = (t >= 0) & (t < self.num_classes)
        p, t = p[mask], t[mask]
        idx = t * self.num_classes + p
        bc = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.mat += bc.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.mat)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return iou

    def per_class_dice(self) -> np.ndarray:
        tp = np.diag(self.mat)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        dice = np.where(denom > 0, (2 * tp) / np.maximum(denom, 1), np.nan)
        return dice

    def mean_iou(self) -> float:
        return float(np.nanmean(self.per_class_iou()))

    def mean_dice(self) -> float:
        return float(np.nanmean(self.per_class_dice()))
