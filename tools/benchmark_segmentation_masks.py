"""Evaluate weak segmentation rules against VIA polygon masks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from resistor_detector.advanced import segment_resistor_mask

ROOT = Path(__file__).resolve().parents[1]


def _load_via(annotation_path: Path, image_dir: Path) -> list[tuple[str, np.ndarray]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    records: list[tuple[str, np.ndarray]] = []
    for item in payload.values():
        filename = item.get("filename")
        if not filename:
            continue
        image = cv2.imread(str(image_dir / filename))
        if image is None:
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in item.get("regions", []):
            shape = region.get("shape_attributes", {})
            xs = shape.get("all_points_x")
            ys = shape.get("all_points_y")
            if xs and ys and len(xs) == len(ys):
                polygon = np.column_stack((xs, ys)).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 255)
            elif shape.get("name") == "rect":
                x, y = int(shape["x"]), int(shape["y"])
                w, h = int(shape["width"]), int(shape["height"])
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        records.append((filename, mask))
    return records


def _metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    pred = predicted > 0
    truth = expected > 0
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "predicted_fraction": float(np.mean(pred)),
        "truth_fraction": float(np.mean(truth)),
    }


def run(annotation: Path, image_dir: Path, output_dir: Path) -> dict[str, object]:
    records = _load_via(annotation, image_dir)
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        f"rule-sat-{threshold}": lambda frame, threshold=threshold: segment_resistor_mask(
            frame, saturation_min=threshold
        )
        for threshold in (20, 35, 50, 70, 90)
    }
    methods.update(
        {
            f"saturation-{threshold}": lambda frame, threshold=threshold: segment_resistor_mask(
                frame, saturation_min=threshold, use_contrast=False
            )
            for threshold in (20, 35, 50, 70, 90)
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    for method, segment in methods.items():
        started = time.perf_counter()
        per_image: list[dict[str, float]] = []
        for filename, expected in records:
            image = cv2.imread(str(image_dir / filename))
            if image is None:
                continue
            per_image.append(_metrics(segment(image), expected))
        summary = {
            "method": method,
            "images": len(per_image),
            "iou_mean": statistics.mean(row["iou"] for row in per_image),
            "dice_mean": statistics.mean(row["dice"] for row in per_image),
            "precision_mean": statistics.mean(row["precision"] for row in per_image),
            "recall_mean": statistics.mean(row["recall"] for row in per_image),
            "elapsed_seconds": time.perf_counter() - started,
        }
        summaries[method] = summary
        print(
            f"DONE {method} iou={summary['iou_mean']:.4f} dice={summary['dice_mean']:.4f}",
            flush=True,
        )
    ranked = sorted(summaries.values(), key=lambda row: row["dice_mean"], reverse=True)
    result = {
        "annotation": str(annotation),
        "image_dir": str(image_dir),
        "images": len(records),
        "best_method": ranked[0] if ranked else None,
        "methods": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.annotation, args.image_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
