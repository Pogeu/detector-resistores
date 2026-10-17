"""Reprodução do baseline CNN descrito no artigo de Caoile (2026).

O artigo descreve um classificador de imagem, não um detector de objetos:
o resistor deve estar dentro de uma região-guia e a saída é uma classe de
valor em ohms. Esta implementação preserva esse fluxo e deixa explícitos os
parâmetros que o artigo não informa, como filtros e intensidade das
augmentations.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


IMAGE_SIZE = 64
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 3
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ResistorImage:
    path: Path
    value_token: str
    value_ohms: float
    band_count: int
    source_class: str


def parse_value_token(token: str) -> float:
    """Converte notação de resistor como 4K7, 1K2, 100R e 1M."""

    normalized = token.strip().upper().replace("Ω", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([RKM])([0-9]*)", normalized)
    if match:
        left, unit, right = match.groups()
        number = float(left + (f".{right}" if right else ""))
    else:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([RKM])?", normalized)
        if not match:
            raise ValueError(f"valor de resistor inválido: {token}")
        number = float(match.group(1))
        unit = match.group(2) or "R"

    multiplier = {"R": 1.0, "K": 1_000.0, "M": 1_000_000.0}[unit]
    return number * multiplier


def _require_torch() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "A reprodução CNN requer torch, torchvision e Pillow. "
            "Instale as dependências opcionais com `pip install -e .[cnn]`."
        ) from exc
    return torch, Image, nn, DataLoader, (Dataset, transforms)


def discover_images(dataset_root: Path) -> list[ResistorImage]:
    """Lê a estrutura do Barrettotte e agrupa rótulos pelo valor elétrico."""

    root = dataset_root / "extracted" if (dataset_root / "extracted").is_dir() else dataset_root
    records: list[ResistorImage] = []
    pattern = re.compile(r"^(?P<bands>[45])B-(?P<value>.+?)-T[0-9]+(?:\.[0-9]+)?$")
    for image_path in sorted(root.rglob("*.jpg")):
        parts = image_path.relative_to(root).parts
        if len(parts) < 3:
            continue
        match = pattern.match(parts[-2])
        if not match:
            continue
        token = match.group("value")
        records.append(
            ResistorImage(
                path=image_path,
                value_token=token,
                value_ohms=parse_value_token(token),
                band_count=int(match.group("bands")),
                source_class=parts[-2],
            )
        )
    if not records:
        raise FileNotFoundError(f"nenhuma imagem Barrettotte encontrada em {dataset_root}")
    return records


def _stratified_split(
    records: Sequence[ResistorImage],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ResistorImage], list[ResistorImage]]:
    rng = random.Random(seed)
    by_label: dict[str, list[ResistorImage]] = {}
    for record in records:
        by_label.setdefault(record.value_token, []).append(record)

    train: list[ResistorImage] = []
    validation: list[ResistorImage] = []
    for label_records in by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train.extend(shuffled)
            continue
        count = max(1, round(len(shuffled) * validation_fraction))
        count = min(count, len(shuffled) - 1)
        validation.extend(shuffled[:count])
        train.extend(shuffled[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _random_split(
    records: Sequence[ResistorImage],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ResistorImage], list[ResistorImage]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - validation_fraction))
    return shuffled[:cut], shuffled[cut:]


def split_records(
    records: Sequence[ResistorImage],
    validation_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
    strategy: str = "stratified",
) -> tuple[list[ResistorImage], list[ResistorImage]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    if strategy == "stratified":
        return _stratified_split(records, validation_fraction, seed)
    if strategy == "random":
        return _random_split(records, validation_fraction, seed)
    raise ValueError(f"estratégia de divisão desconhecida: {strategy}")


class _ResistorDataset:
    def __init__(self, records: Sequence[ResistorImage], labels: dict[str, int], transform: Any) -> None:
        torch, image_type, _, _, _ = _require_torch()
        self._torch = torch
        self._image_type = image_type
        self.records = list(records)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        record = self.records[index]
        image = self._image_type.open(record.path).convert("RGB")
        return self.transform(image), self.labels[record.value_token]


def build_model(class_count: int, first_filters: int = 32, second_filters: int = 64) -> Any:
    """CNN leve: duas convoluções, pooling 2x2, densa 128 e logits."""

    _, _, nn, _, _ = _require_torch()
    return nn.Sequential(
        nn.Conv2d(3, first_filters, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(first_filters, second_filters, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(second_filters * 16 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, class_count),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _, _, _ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accuracy(model: Any, loader: Any, device: Any) -> float:
    torch, _, _, _, _ = _require_torch()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += int((logits.argmax(dim=1).cpu() == labels).sum())
            total += len(labels)
    return correct / total if total else 0.0


def train_model(
    dataset_root: Path,
    output_path: Path,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    validation_fraction: float = 0.2,
    split_strategy: str = "stratified",
    seed: int = DEFAULT_SEED,
    rotation_degrees: float = 15.0,
    brightness_delta: float = 0.2,
    center_crop_size: int | None = None,
    learning_rate: float = 1e-3,
    first_filters: int = 32,
    second_filters: int = 64,
) -> dict[str, Any]:
    torch, _, _, DataLoader, (_, transforms) = _require_torch()
    _seed_everything(seed)
    records = discover_images(dataset_root)
    labels_by_token = {token: i for i, token in enumerate(sorted({r.value_token for r in records}, key=parse_value_token))}
    train_records, validation_records = split_records(
        records, validation_fraction, seed, split_strategy
    )

    train_operations: list[Any] = []
    validation_operations: list[Any] = []
    if center_crop_size is not None:
        train_operations.append(transforms.CenterCrop(center_crop_size))
        validation_operations.append(transforms.CenterCrop(center_crop_size))
    train_operations.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(rotation_degrees),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=brightness_delta),
            transforms.ToTensor(),
        ]
    )
    validation_operations.extend(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    train_transform = transforms.Compose(train_operations)
    validation_transform = transforms.Compose(validation_operations)
    train_loader = DataLoader(
        _ResistorDataset(train_records, labels_by_token, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _ResistorDataset(validation_records, labels_by_token, validation_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels_by_token), first_filters, second_filters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_accuracy = _accuracy(model, validation_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            stale_epochs = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_index": labels_by_token,
                    "class_values_ohms": {
                        token: parse_value_token(token) for token in labels_by_token
                    },
                    "model": {
                        "image_size": IMAGE_SIZE,
                        "first_filters": first_filters,
                        "second_filters": second_filters,
                    },
                    "protocol": {
                        "rotation_degrees": rotation_degrees,
                        "brightness_delta": brightness_delta,
                        "center_crop_size": center_crop_size,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "split_strategy": split_strategy,
                        "validation_fraction": validation_fraction,
                        "seed": seed,
                    },
                },
                output_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - started
    summary = {
        "dataset_root": str(dataset_root),
        "image_count": len(records),
        "class_count": len(labels_by_token),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "elapsed_seconds": elapsed,
        "checkpoint": str(output_path),
        "history": history,
    }
    return summary


def predict_image(checkpoint_path: Path, image_path: Path) -> dict[str, Any]:
    torch, image_type, _, _, (_, transforms) = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_index = checkpoint["class_to_index"]
    inverse = {index: token for token, index in class_to_index.items()}
    config = checkpoint["model"]
    model = build_model(
        len(class_to_index), config["first_filters"], config["second_filters"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    operations: list[Any] = []
    center_crop_size = checkpoint.get("protocol", {}).get("center_crop_size")
    if center_crop_size is not None:
        operations.append(transforms.CenterCrop(center_crop_size))
    operations.extend([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    transform = transforms.Compose(operations)
    image = transform(image_type.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]
    index = int(probabilities.argmax())
    token = inverse[index]
    return {
        "value_token": token,
        "value_ohms": checkpoint["class_values_ohms"][token],
        "confidence": float(probabilities[index]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="treina o baseline CNN do artigo")
    train.add_argument("--dataset", type=Path, default=Path("data/sources/kaggle-barrettotte"))
    train.add_argument("--output", type=Path, default=Path("data/results/caoile-cnn.pt"))
    train.add_argument("--metrics", type=Path, default=Path("data/results/caoile-cnn-metrics.json"))
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    train.add_argument("--split", choices=("stratified", "random"), default="stratified")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--rotation-degrees", type=float, default=15.0)
    train.add_argument("--brightness-delta", type=float, default=0.2)
    train.add_argument(
        "--center-crop-size",
        type=int,
        default=None,
        help="recorta um quadrado antes do resize; 720 é adequado às imagens 1280x720",
    )
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--first-filters", type=int, default=32)
    train.add_argument("--second-filters", type=int, default=64)
    predict = subparsers.add_parser("predict", help="classifica uma imagem com um checkpoint")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("image", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        summary = train_model(
            args.dataset,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            split_strategy=args.split,
            seed=args.seed,
            rotation_degrees=args.rotation_degrees,
            brightness_delta=args.brightness_delta,
            center_crop_size=args.center_crop_size,
            learning_rate=args.learning_rate,
            first_filters=args.first_filters,
            second_filters=args.second_filters,
        )
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
        return 0
    result = predict_image(args.checkpoint, args.image)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
