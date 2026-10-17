"""Converte exports YOLO do Roboflow em manifests do benchmark local."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources"
OUTPUT = ROOT / "data" / "manifests"

DATASETS = {
    "resistordataset": "roboflow-resistordataset",
    "electronic-ezdmf": "roboflow-electronic-ezdmf",
    "electronic-components": "roboflow-electronic-components",
    "all-components": "roboflow-all-components",
}


def _class_names(path: Path) -> list[str]:
    names_line = next(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("names:")
    )
    values = names_line.split(":", 1)[1].strip()
    parsed = ast.literal_eval(values)
    if isinstance(parsed, list):
        return [str(value) for value in parsed]
    raise ValueError(f"names inválido em {path}")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def _label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _records_for_dataset(dataset_id: str, source_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    extracted = source_dir / "extracted"
    names = _class_names(extracted / "data.yaml")
    records: list[dict[str, Any]] = []
    for split_dir in ("train", "valid", "test"):
        images_dir = extracted / split_dir / "images"
        if not images_dir.is_dir():
            continue
        first_image = next(images_dir.rglob("*.jpg"), None)
        if first_image is None:
            continue
        # Os exports Roboflow têm resolução fixa por split; evitar abrir
        # milhares de JPEGs sincronizados pelo OneDrive individualmente.
        width, height = _image_size(first_image)
        for image_path in sorted(images_dir.rglob("*.jpg")):
            objects: list[dict[str, Any]] = []
            label_path = _label_path(image_path)
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 5:
                        continue
                    class_id, cx, cy, box_width, box_height = fields
                    class_index = int(class_id)
                    class_name = names[class_index]
                    if class_name.lower() != "resistor":
                        continue
                    cx, cy = float(cx), float(cy)
                    box_width, box_height = float(box_width), float(box_height)
                    objects.append(
                        {
                            "bbox": [
                                (cx - box_width / 2) * width,
                                (cy - box_height / 2) * height,
                                box_width * width,
                                box_height * height,
                            ],
                            "class_id": class_index,
                            "class_name": class_name,
                            "bands": [],
                        }
                    )
            if not objects:
                continue
            records.append(
                {
                    "id": f"{dataset_id}-{split_dir}-{image_path.stem}",
                    "source_id": dataset_id,
                    "path": image_path.relative_to(ROOT).as_posix(),
                    "media_type": "image",
                    "split": split_dir,
                    "task": "detection",
                    "objects": objects,
                }
            )
    return records, names


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset_id, source_name in DATASETS.items():
        records, names = _records_for_dataset(dataset_id, SOURCES / source_name)
        output_path = OUTPUT / f"{dataset_id}-resistor-detection.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        object_count = sum(len(record["objects"]) for record in records)
        print(
            f"{output_path.relative_to(ROOT)}: images={len(records)} "
            f"objects={object_count} classes={names}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
