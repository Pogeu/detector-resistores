"""Converte os três datasets Kaggle baixados para o formato do benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources"
OUTPUT = ROOT / "data" / "manifests"
OUTPUT.mkdir(parents=True, exist_ok=True)


def parse_resistance(token: str) -> float:
    normalized = token.strip().upper().replace("Ω", "")
    normalized = re.sub(r"OHMS?$", "", normalized)
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([RKM]?)", normalized)
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
    else:
        embedded = re.fullmatch(r"([0-9]+)([RKM])([0-9]+)", normalized)
        if embedded:
            number = float(f"{embedded.group(1)}.{embedded.group(3)}")
            suffix = embedded.group(2)
        else:
            mixed = re.fullmatch(r"([0-9]+)[RKM]([0-9]+)([KM])", normalized)
            if not mixed:
                raise ValueError(f"valor de resistor não reconhecido: {token}")
            number = float(f"{mixed.group(1)}.{mixed.group(2)}")
            suffix = mixed.group(3)
    multiplier = {"": 1.0, "R": 1.0, "K": 1_000.0, "M": 1_000_000.0}[suffix]
    value = number * multiplier
    return int(value) if value.is_integer() else value


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"imagem inválida: {path}")
    height, width = image.shape[:2]
    return width, height


def full_image_bbox(path: Path) -> list[int]:
    width, height = image_size(path)
    return [0, 0, width, height]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(records)} amostras")


def build_crop_value_manifest(
    dataset_dir: Path,
    output_name: str,
    *,
    use_barrettotte_bands: bool = False,
) -> None:
    records: list[dict[str, Any]] = []
    color_maps: dict[str, dict[str, list[str]]] = {}
    if use_barrettotte_bands:
        for band_count in (4, 5):
            json_path = dataset_dir / "extracted" / f"{band_count}-band" / f"{band_count}-band.json"
            color_maps[str(band_count)] = json.loads(json_path.read_text(encoding="utf-8"))

    for image_path in sorted((dataset_dir / "extracted").rglob("*.jpg")):
        relative = image_path.relative_to(dataset_dir / "extracted")
        if use_barrettotte_bands:
            band_dir, class_dir = relative.parts[:2]
            band_count = int(band_dir.split("-")[0])
            parts = class_dir.split("-")
            value = parse_resistance(parts[1])
            tolerance_match = re.search(r"T([0-9]+(?:\.[0-9]+)?)", class_dir.upper())
            tolerance = float(tolerance_match.group(1)) if tolerance_match else None
            bands = color_maps[str(band_count)].get(str(int(value)), [])
            source = "barrettotte"
        else:
            class_dir = relative.parts[0]
            value_token = re.split(r"[_-]", class_dir, maxsplit=1)[0]
            value = parse_resistance(value_token)
            tolerance = None
            bands = []
            source = "eralpozcan"

        records.append(
            {
                "id": f"{source}-{image_path.stem}",
                "source_id": source,
                "path": image_path.relative_to(ROOT).as_posix(),
                "media_type": "image",
                "task": "crop_value",
                "objects": [
                    {
                        "bbox": full_image_bbox(image_path),
                        "bands": bands,
                        "ohms": value,
                        "tolerance_percent": tolerance,
                    }
                ],
            }
        )
    write_records(OUTPUT / output_name, records)


def build_segmentation_manifest() -> None:
    dataset_dir = SOURCES / "kaggle-alptugaydin" / "extracted"
    annotations = json.loads((dataset_dir / "train.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for key, annotation in annotations.items():
        filename = annotation["filename"]
        matches = list(dataset_dir.rglob(filename))
        if not matches:
            continue
        objects = []
        for region in annotation.get("regions", []):
            shape = region.get("shape_attributes", {})
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if not xs or not ys:
                continue
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            objects.append(
                {
                    "bbox": [left, top, right - left, bottom - top],
                    "bands": [],
                }
            )
        if objects:
            records.append(
                {
                    "id": f"alptugaydin-{filename}",
                    "source_id": "alptugaydin",
                    "path": matches[0].relative_to(ROOT).as_posix(),
                    "media_type": "image",
                    "task": "segmentation",
                    "objects": objects,
                }
            )
    write_records(OUTPUT / "alptugaydin-segmentation.jsonl", records)


if __name__ == "__main__":
    build_crop_value_manifest(
        SOURCES / "kaggle-eralpozcan",
        "eralpozcan-crop-value.jsonl",
    )
    build_crop_value_manifest(
        SOURCES / "kaggle-barrettotte",
        "barrettotte-crop-value.jsonl",
        use_barrettotte_bands=True,
    )
    build_segmentation_manifest()
