"""Train a UNet for CityScape semantic segmentation (23 classes).

- 80/20 train/test split with seed 42
- Cross-entropy loss, Adam optimizer
- Tracks per-epoch train loss, train mIoU, train mDice, test mIoU, test mDice
- Saves best model (by test mIoU) + per-epoch metrics JSON + plots
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import NUM_CLASSES, make_loaders
from metrics import ConfusionMatrix
from model import UNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, num_classes: int):
    model.eval()
    cm = ConfusionMatrix(num_classes)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        cm.update(pred, y)
    return cm.mean_iou(), cm.mean_dice()


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes: int):
    model.train()
    total_loss = 0.0
    n_batches = 0
    cm = ConfusionMatrix(num_classes)
    pbar = tqdm(loader, leave=False, desc="train")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        with torch.no_grad():
            cm.update(logits.argmax(dim=1), y)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, n_batches), cm.mean_iou(), cm.mean_dice()


def plot_curves(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], marker="o", color="#c0392b")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_miou"], label="train mIoU", marker="o")
    plt.plot(epochs, history["test_miou"], label="test mIoU", marker="s")
    plt.title("Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "miou_curve.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_mdice"], label="train mDice", marker="o", color="#2e86de")
    plt.plot(epochs, history["test_mdice"], label="test mDice", marker="s", color="#e67e22")
    plt.title("Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("mDice")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mdice_curve.png", dpi=130)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    train_loader, test_loader, train_pairs, test_pairs = make_loaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"[train] train={len(train_pairs)} test={len(test_pairs)}")

    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=32).to(device)
    print(f"[train] params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_miou": [],
        "train_mdice": [],
        "test_miou": [],
        "test_mdice": [],
    }

    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_miou, train_mdice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, NUM_CLASSES
        )
        test_miou, test_mdice = evaluate(model, test_loader, device, NUM_CLASSES)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["train_mdice"].append(train_mdice)
        history["test_miou"].append(test_miou)
        history["test_mdice"].append(test_mdice)

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train mIoU={train_miou:.4f} mDice={train_mdice:.4f} | "
            f"test mIoU={test_miou:.4f} mDice={test_mdice:.4f} | "
            f"{dt:.1f}s",
            flush=True,
        )

        plot_curves(history, out_dir)
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if test_miou > best_miou:
            best_miou = test_miou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_miou": test_miou,
                    "test_mdice": test_mdice,
                    "args": vars(args),
                },
                out_dir / "best_model.pt",
            )
            print(f"[train] saved new best model (mIoU={test_miou:.4f})")

    final = {
        "best_test_miou": max(history["test_miou"]),
        "best_test_mdice_at_best_miou": history["test_mdice"][
            int(np.argmax(history["test_miou"]))
        ],
        "final_test_miou": history["test_miou"][-1],
        "final_test_mdice": history["test_mdice"][-1],
        "epochs": args.epochs,
    }
    with (out_dir / "final_metrics.json").open("w") as f:
        json.dump(final, f, indent=2)
    print("[train] DONE")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
