"""Evaluate one-class detector by source on the aggregate manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, aw, ah = left
    bx1, by1, bw, bh = right
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def evaluate(model_path: Path, manifest_path: Path, output_path: Path, split: str, device: str, imgsz: int) -> dict[str, Any]:
    from ultralytics import YOLO

    records = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_split = record.get("aggregate_split", record.get("split", "train"))
        if record_split in {split, "valid" if split == "val" else split}:
            records.append(record)
    model = YOLO(str(model_path))
    counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    paths = [ROOT / record["path"] for record in records]
    for start in range(0, len(records), 32):
        batch_records = records[start : start + 32]
        results = model.predict(
            source=[str(path) for path in paths[start : start + 32]],
            imgsz=imgsz,
            device=device,
            batch=32,
            verbose=False,
            conf=0.25,
        )
        for record, result in zip(batch_records, results):
            source = str(record.get("source_id", "unknown"))
            targets = [obj["bbox"] for obj in record.get("objects", []) if obj.get("bbox")]
            predictions = []
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().tolist():
                    predictions.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
            matched: set[int] = set()
            true_positive = 0
            for prediction in predictions:
                matches = [(index, iou(prediction, target)) for index, target in enumerate(targets) if index not in matched]
                if matches:
                    index, score = max(matches, key=lambda item: item[1])
                    if score >= 0.5:
                        matched.add(index)
                        true_positive += 1
            counters[source]["images"] += 1
            counters[source]["tp"] += true_positive
            counters[source]["fp"] += len(predictions) - true_positive
            counters[source]["fn"] += len(targets) - true_positive

    per_source = {}
    totals = defaultdict(float)
    for source, values in sorted(counters.items()):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_source[source] = {
            **{key: int(value) for key, value in values.items() if key in {"images", "tp", "fp", "fn"}},
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
        for key in ("images", "tp", "fp", "fn"):
            totals[key] += values[key]
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "model": str(model_path),
        "manifest": str(manifest_path),
        "split": split,
        "images": int(totals["images"]),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_source": per_source,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    evaluate(args.model, args.manifest, args.output, args.split, args.device, args.imgsz)


if __name__ == "__main__":
    main()
