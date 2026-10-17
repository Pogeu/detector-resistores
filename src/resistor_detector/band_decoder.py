"""Multi-head resistor-band decoder trained from value-labelled images.

The Barrettotte folders encode resistance and tolerance, so the electrical
value can be converted into the expected 4/5 colour sequence.  This keeps the
learning target compositional instead of using one class for every ohmic
value.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .caoile_cnn import ResistorImage, discover_images, parse_value_token, split_records
from .colors import DIGITS, MULTIPLIERS, TOLERANCES, decode_color_bands

IMAGE_SIZE = 128
COLOR_NAMES = tuple(DIGITS) + ("gold", "silver")
COLOR_TO_INDEX = {name: index for index, name in enumerate(COLOR_NAMES)}
INDEX_TO_COLOR = {index: name for name, index in COLOR_TO_INDEX.items()}
TOLERANCE_TO_COLOR = {1.0: "brown", 2.0: "red", 0.5: "green", 0.25: "blue", 0.1: "violet", 0.05: "gray", 5.0: "gold", 10.0: "silver"}


@dataclass(frozen=True)
class BandLabel:
    colors: tuple[str, ...]
    band_count: int
    value_ohms: float


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("instale as dependencias opcionais: pip install -e '.[cnn]'") from exc
    return torch, Image, nn, transforms


def _tolerance_from_source(source_class: str) -> float:
    match = re.search(r"-T(?P<tolerance>[0-9.]+)$", source_class)
    return float(match.group("tolerance")) if match else 5.0


def _digit_color(value: int) -> str:
    return next(name for name, digit in DIGITS.items() if digit == value)


def _multiplier_color(exponent: int) -> str:
    return next(name for name, power in MULTIPLIERS.items() if power == exponent)


def value_to_bands(value_ohms: float, band_count: int, tolerance_percent: float) -> BandLabel:
    """Convert a nominal value and tolerance to its IEC colour sequence."""

    significant_digits = 2 if band_count == 4 else 3
    exponent = int(math.floor(math.log10(value_ohms))) - significant_digits + 1
    significant = int(round(value_ohms / (10**exponent)))
    if significant >= 10**significant_digits:
        significant //= 10
        exponent += 1
    digits = [int(char) for char in str(significant).zfill(significant_digits)]
    colors = tuple(_digit_color(digit) for digit in digits)
    colors += (_multiplier_color(exponent), TOLERANCE_TO_COLOR.get(tolerance_percent, "gold"))
    return BandLabel(colors=colors, band_count=band_count, value_ohms=value_ohms)


def labels_for_records(records: Sequence[ResistorImage]) -> list[BandLabel]:
    return [
        value_to_bands(record.value_ohms, record.band_count, _tolerance_from_source(record.source_class))
        for record in records
    ]


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(torch: Any, nn: Any) -> Any:
    class BandDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 192, 3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.shared = nn.Sequential(nn.Flatten(), nn.Linear(192, 128), nn.ReLU(inplace=True), nn.Dropout(0.2))
            self.band_heads = nn.ModuleList(nn.Linear(128, len(COLOR_NAMES)) for _ in range(5))
            self.count_head = nn.Linear(128, 2)

        def forward(self, images: Any) -> tuple[Any, Any]:
            shared = self.shared(self.features(images))
            return tuple(head(shared) for head in self.band_heads), self.count_head(shared)

    return BandDecoder()


def _dataset(records: Sequence[ResistorImage], labels: Sequence[BandLabel], transform: Any, Image: Any, torch: Any) -> Any:
    class BandDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
            with Image.open(records[index].path) as image:
                image = image.convert("RGB")
                tensor = transform(image)
            label = labels[index]
            padded = [COLOR_TO_INDEX[color] for color in label.colors]
            padded.extend([0] * (5 - len(padded)))
            return tensor, torch.tensor(padded, dtype=torch.long), torch.tensor(label.band_count - 4, dtype=torch.long)

    return BandDataset()


def _accuracy(model: Any, loader: Any, device: Any, torch: Any, labels: Sequence[BandLabel]) -> dict[str, float]:
    model.eval()
    correct_sequences = 0
    correct_values = 0
    total = 0
    with torch.no_grad():
        offset = 0
        for images, targets, count_targets in loader:
            images = images.to(device)
            band_logits, count_logits = model(images)
            predicted_count = count_logits.argmax(1).cpu().tolist()
            predicted_colors = [logit.argmax(1).cpu().tolist() for logit in band_logits]
            for row in range(len(images)):
                label = labels[offset + row]
                count = 5 if predicted_count[row] else 4
                colors = tuple(INDEX_TO_COLOR[predicted_colors[pos][row]] for pos in range(count))
                target_colors = label.colors
                correct_sequences += int(colors == target_colors)
                decoded = decode_color_bands(colors)
                correct_values += int(decoded is not None and decoded.ohms == label.value_ohms)
                total += 1
            offset += len(images)
    return {
        "sequence_accuracy": correct_sequences / total if total else 0.0,
        "value_accuracy": correct_values / total if total else 0.0,
        "count_accuracy": 0.0,
    }


def train_model(
    dataset_root: Path,
    output_path: Path,
    metrics_path: Path,
    *,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 10,
    seed: int = 20260920,
    center_crop_size: int | None = 720,
) -> dict[str, Any]:
    torch, Image, nn, transforms = _require_torch()
    _seed_everything(seed, torch)
    records = discover_images(dataset_root)
    labels = labels_for_records(records)
    train_records, validation_records = split_records(records, 0.2, seed, "stratified")
    record_to_label = {record.path: label for record, label in zip(records, labels)}
    train_labels = [record_to_label[record.path] for record in train_records]
    validation_labels = [record_to_label[record.path] for record in validation_records]

    train_ops: list[Any] = []
    validation_ops: list[Any] = []
    if center_crop_size is not None:
        train_ops.append(transforms.CenterCrop(center_crop_size))
        validation_ops.append(transforms.CenterCrop(center_crop_size))
    train_ops += [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomRotation(12), transforms.ColorJitter(brightness=0.2, contrast=0.15), transforms.ToTensor()]
    validation_ops += [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    train_loader = torch.utils.data.DataLoader(_dataset(train_records, train_labels, transforms.Compose(train_ops), Image, torch), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(_dataset(validation_records, validation_labels, transforms.Compose(validation_ops), Image, torch), batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(torch, nn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_sequence = -1.0
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets, count_targets in train_loader:
            images, targets, count_targets = images.to(device), targets.to(device), count_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            band_logits, count_logits = model(images)
            loss = loss_function(count_logits, count_targets)
            for position, logits in enumerate(band_logits):
                loss = loss + loss_function(logits, targets[:, position])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _accuracy(model, validation_loader, device, torch, validation_labels)
        metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        history.append(metrics)
        print(f"epoch={epoch:02d} loss={metrics['train_loss']:.4f} sequence={metrics['sequence_accuracy']:.4f} value={metrics['value_accuracy']:.4f}", flush=True)
        if metrics["sequence_accuracy"] > best_sequence:
            best_sequence = metrics["sequence_accuracy"]
            stale = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model": {"image_size": IMAGE_SIZE}, "colors": COLOR_NAMES, "center_crop_size": center_crop_size}, output_path)
        else:
            stale += 1
            if stale >= patience:
                break
    summary = {"dataset_root": str(dataset_root), "image_count": len(records), "train_count": len(train_records), "validation_count": len(validation_records), "device": str(device), "epochs_requested": epochs, "epochs_completed": len(history), "best_sequence_accuracy": best_sequence, "best_value_accuracy": max((row["value_accuracy"] for row in history), default=0.0), "elapsed_seconds": time.perf_counter() - started, "checkpoint": str(output_path), "history": history}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/sources/kaggle-barrettotte"))
    parser.add_argument("--output", type=Path, default=Path("data/results/band-decoder.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("data/results/band-decoder-metrics.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--center-crop-size", type=int, default=720)
    args = parser.parse_args()
    train_model(args.dataset, args.output, args.metrics, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, seed=args.seed, center_crop_size=args.center_crop_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
