"""Convert VIA polygon annotations to a portable YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


def _polygon(shape: dict, width: int, height: int) -> list[float]:
    if shape.get("all_points_x") and shape.get("all_points_y"):
        points = list(zip(shape["all_points_x"], shape["all_points_y"]))
    elif shape.get("name") == "rect":
        x, y = float(shape["x"]), float(shape["y"])
        w, h = float(shape["width"]), float(shape["height"])
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        return []
    normalized = []
    for x, y in points:
        normalized.extend((max(0.0, min(1.0, float(x) / width)), max(0.0, min(1.0, float(y) / height))))
    return normalized if len(normalized) >= 6 else []


def prepare(annotation: Path, image_dir: Path, output_dir: Path, val_fraction: float, seed: int) -> dict:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    records = []
    for item in payload.values():
        image_path = image_dir / item["filename"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        polygons = [_polygon(region.get("shape_attributes", {}), width, height) for region in item.get("regions", [])]
        polygons = [polygon for polygon in polygons if polygon]
        if polygons:
            records.append((image_path, polygons))
    random.Random(seed).shuffle(records)
    val_count = max(1, round(len(records) * val_fraction))
    splits = {"val": records[:val_count], "train": records[val_count:]}
    for split, split_records in splits.items():
        image_out = output_dir / "images" / split
        label_out = output_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image_path, polygons in split_records:
            shutil.copy2(image_path, image_out / image_path.name)
            label_path = label_out / (image_path.stem + ".txt")
            label_path.write_text(
                "\n".join("0 " + " ".join(f"{value:.8f}" for value in polygon) for polygon in polygons) + "\n",
                encoding="utf-8",
            )
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: resistor\n",
        encoding="utf-8",
    )
    summary = {
        "annotation": str(annotation),
        "images": len(records),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "seed": seed,
        "yaml": str(yaml_path),
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotation, args.image_dir, args.output_dir, args.val_fraction, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

