"""Train a small U-Net on the VIA polygon masks from Alptugaydin."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from benchmark_segmentation_masks import _load_via

ROOT = Path(__file__).resolve().parents[1]


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("instale as dependencias CNN antes do treino") from exc
    return torch, nn


def _seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _conv_block(nn: Any, input_channels: int, output_channels: int) -> Any:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


def _build_model(nn: Any) -> Any:
    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = _conv_block(nn, 3, 32)
            self.down2 = _conv_block(nn, 32, 64)
            self.down3 = _conv_block(nn, 64, 128)
            self.bridge = _conv_block(nn, 128, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = _conv_block(nn, 256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = _conv_block(nn, 128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = _conv_block(nn, 64, 32)
            self.output = nn.Conv2d(32, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, image: Any) -> Any:
            first = self.down1(image)
            second = self.down2(self.pool(first))
            third = self.down3(self.pool(second))
            bridge = self.bridge(self.pool(third))
            decoded = self.up3(bridge)
            decoded = self.dec3(torch.cat((decoded, third), dim=1))
            decoded = self.up2(decoded)
            decoded = self.dec2(torch.cat((decoded, second), dim=1))
            decoded = self.up1(decoded)
            decoded = self.dec1(torch.cat((decoded, first), dim=1))
            return self.output(decoded)

    return TinyUNet()


class _MaskDataset:
    def __init__(self, records: list[tuple[str, np.ndarray]], image_dir: Path, size: int, augment: bool) -> None:
        self.records = records
        self.image_dir = image_dir
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        filename, mask = self.records[index]
        image = cv2.imread(str(self.image_dir / filename))
        if image is None:
            raise FileNotFoundError(self.image_dir / filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment and random.random() < 0.25:
            image = np.clip(image.astype(np.float32) * random.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
        image_array = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        mask_array = (mask > 0).astype(np.float32)[None, ...]
        return image_array, mask_array


def _batch(dataset: _MaskDataset, indices: list[int], torch: Any, batch_size: int, shuffle: bool) -> Any:
    order = indices[:]
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        current = order[start : start + batch_size]
        images, masks = zip(*(dataset[index] for index in current))
        yield torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


def _metrics(logits: Any, masks: Any, torch: Any) -> tuple[float, float, float, float]:
    predicted = logits.sigmoid() > 0.5
    truth = masks > 0.5
    tp = (predicted & truth).sum().item()
    fp = (predicted & ~truth).sum().item()
    fn = (~predicted & truth).sum().item()
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return iou, dice, precision, recall


def train(annotation: Path, image_dir: Path, output: Path, metrics_path: Path, *, size: int = 512, epochs: int = 30, batch_size: int = 4, patience: int = 8, seed: int = 20260920) -> dict[str, Any]:
    torch, nn = _require_torch()
    _seed(seed, torch)
    records = _load_via(annotation, image_dir)
    random.Random(seed).shuffle(records)
    split = max(1, int(len(records) * 0.8))
    train_records, validation_records = records[:split], records[split:]
    train_dataset = _MaskDataset(train_records, image_dir, size, True)
    validation_dataset = _MaskDataset(validation_records, image_dir, size, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(nn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_dice = -1.0
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, masks in _batch(train_dataset, list(range(len(train_dataset))), torch, batch_size, True):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
            probabilities = logits.sigmoid()
            dice_loss = 1 - (2 * (probabilities * masks).sum() + 1) / (probabilities.sum() + masks.sum() + 1)
            loss = bce + dice_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        values: list[tuple[float, float, float, float]] = []
        with torch.no_grad():
            for images, masks in _batch(validation_dataset, list(range(len(validation_dataset))), torch, batch_size, False):
                values.append(_metrics(model(images.to(device)), masks.to(device), torch))
        current = {
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "iou": float(np.mean([row[0] for row in values])) if values else 0.0,
            "dice": float(np.mean([row[1] for row in values])) if values else 0.0,
            "precision": float(np.mean([row[2] for row in values])) if values else 0.0,
            "recall": float(np.mean([row[3] for row in values])) if values else 0.0,
        }
        history.append(current)
        print(f"epoch={epoch:02d} loss={current['train_loss']:.4f} dice={current['dice']:.4f} iou={current['iou']:.4f}", flush=True)
        if current["dice"] > best_dice:
            best_dice = current["dice"]
            stale = 0
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "size": size}, output)
        else:
            stale += 1
            if stale >= patience:
                break
    summary = {"annotation": str(annotation), "images": len(records), "train_images": len(train_records), "validation_images": len(validation_records), "device": str(device), "size": size, "epochs_requested": epochs, "epochs_completed": len(history), "best_dice": best_dice, "best_iou": max((row["iou"] for row in history), default=0.0), "elapsed_seconds": time.perf_counter() - started, "checkpoint": str(output), "history": history}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    train(args.annotation, args.image_dir, args.output, args.metrics, size=args.size, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
