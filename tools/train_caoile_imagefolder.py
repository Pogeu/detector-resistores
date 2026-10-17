"""Train the Caoile-style CNN on a generic class-folder dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from resistor_detector.caoile_cnn import build_model


def _valid_image(path: str) -> bool:
    """Ignore corrupt/HTML masquerading as image files in public exports."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def train(dataset_root: Path, output_path: Path, metrics_path: Path, epochs: int, batch_size: int, seed: int) -> dict:
    import torch
    from PIL import ImageFile
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    classes = datasets.ImageFolder(str(dataset_root), is_valid_file=_valid_image).classes
    base = datasets.ImageFolder(str(dataset_root), is_valid_file=_valid_image)
    by_class: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(base.samples):
        by_class.setdefault(label, []).append(index)
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        count = max(1, round(len(indices) * 0.2))
        count = min(count, len(indices) - 1) if len(indices) > 1 else 0
        val_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = datasets.ImageFolder(str(dataset_root), transform=train_transform, is_valid_file=_valid_image)
    val_data = datasets.ImageFolder(str(dataset_root), transform=val_transform, is_valid_file=_valid_image)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                prediction = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((prediction == labels).sum())
                total += len(labels)
        accuracy = correct / total if total else 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": accuracy})
        print(f"epoch={epoch:02d} train_loss={np.mean(losses):.4f} val_accuracy={accuracy:.4f}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            stale = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": 64}, output_path)
        else:
            stale += 1
            if stale >= 3:
                break
    summary = {
        "dataset": str(dataset_root),
        "classes": len(classes),
        "images": len(base),
        "train_images": len(train_indices),
        "validation_images": len(val_indices),
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output_path),
        "history": history,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset_root, args.output, args.metrics, args.epochs, args.batch, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
