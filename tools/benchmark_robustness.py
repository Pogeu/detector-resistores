"""Evaluate detector robustness under photometric and blur perturbations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from resistor_detector.benchmark import ObjectRecord, SampleRecord, evaluate, load_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG perturbation failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed")
    return decoded


def _conditions() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "original": lambda frame: frame,
        "dim": lambda frame: np.clip(frame.astype(np.float32) * 0.55, 0, 255).astype(np.uint8),
        "bright": lambda frame: np.clip(frame.astype(np.float32) * 1.35, 0, 255).astype(np.uint8),
        "low-contrast": lambda frame: np.clip(frame.astype(np.float32) * 0.65 + 55, 0, 255).astype(np.uint8),
        "blur": lambda frame: cv2.GaussianBlur(frame, (7, 7), 0),
        "jpeg-35": lambda frame: _jpeg(frame, 35),
    }


def _load_samples(manifest: Path, limit: int | None) -> list[SampleRecord]:
    samples = load_jsonl(manifest)
    test_samples = [sample for sample in samples if "-test-" in sample.sample_id]
    selected = test_samples or samples
    return selected[:limit] if limit is not None else selected


def run(
    weights: Path,
    manifest: Path,
    output_dir: Path,
    *,
    limit: int | None = 96,
    imgsz: int = 640,
    confidence: float = 0.1,
    device: str = "0",
) -> dict[str, object]:
    samples = _load_samples(manifest, limit)
    detector = YOLO(str(weights))
    conditions = _conditions()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}

    for condition_name, transform in conditions.items():
        predictions: list[SampleRecord] = []
        started = time.perf_counter()
        for sample in samples:
            if not sample.path:
                continue
            frame = cv2.imread(str(ROOT / sample.path))
            if frame is None:
                raise FileNotFoundError(ROOT / sample.path)
            result = detector.predict(
                source=transform(frame),
                imgsz=imgsz,
                conf=confidence,
                device=device,
                verbose=False,
                save=False,
            )[0]
            boxes: list[tuple[float, float, float, float]] = []
            scores: list[float] = []
            if result.boxes is not None:
                names = detector.names
                for box, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                ):
                    class_name = names[int(class_id)]
                    if str(class_name).lower() != "resistor":
                        continue
                    left, top, right, bottom = box
                    boxes.append((left, top, right - left, bottom - top))
                    scores.append(float(score))
            predictions.append(
                SampleRecord(
                    sample_id=sample.sample_id,
                    source_id=sample.source_id,
                    objects=tuple(
                        ObjectRecord(bbox=box, confidence=score)
                        for box, score in zip(boxes, scores)
                    ),
                    latency_ms=None,
                    strategy=condition_name,
                )
            )
        metrics = evaluate(samples, predictions, strategy=condition_name)
        metrics["elapsed_seconds"] = time.perf_counter() - started
        summaries[condition_name] = metrics
        (output_dir / f"{condition_name}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(
            f"DONE {condition_name} "
            f"f1={metrics['objects']['f1']:.4f} "
            f"recall={metrics['objects']['recall']:.4f}",
            flush=True,
        )

    summary = {
        "weights": str(weights),
        "manifest": str(manifest),
        "samples": len(samples),
        "device": device,
        "conditions": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run(
        args.weights,
        args.manifest,
        args.output_dir,
        limit=args.limit,
        imgsz=args.imgsz,
        confidence=args.confidence,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
