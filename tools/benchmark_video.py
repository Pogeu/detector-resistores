"""Gera e avalia uma sequência de vídeo sintética a partir de uma anotação real."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from resistor_detector.benchmark import ObjectRecord, SampleRecord, evaluate
from resistor_detector.tracking import IoUTracker


ROOT = Path(__file__).resolve().parents[1]


def _read_seed(manifest: Path) -> SampleRecord:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw.get("split") == "test":
            return SampleRecord.from_mapping(raw)
    raise ValueError("manifesto não possui uma amostra test")


def _transform_bbox(bbox: tuple[float, float, float, float], matrix: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y, width, height = bbox
    points = np.float32([[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]])
    transformed = cv2.transform(points, matrix)[0]
    left, top = transformed.min(axis=0)
    right, bottom = transformed.max(axis=0)
    frame_height, frame_width = shape
    left = float(np.clip(left, 0, frame_width - 1))
    top = float(np.clip(top, 0, frame_height - 1))
    right = float(np.clip(right, left + 1, frame_width))
    bottom = float(np.clip(bottom, top + 1, frame_height))
    return left, top, right - left, bottom - top


def run(
    weights: Path,
    manifest: Path,
    output_dir: Path,
    *,
    frames: int = 30,
    imgsz: int = 640,
    confidence: float = 0.1,
    device: str = "0",
) -> dict[str, Any]:
    seed = _read_seed(manifest)
    image_path = ROOT / (seed.path or "")
    source = cv2.imread(str(image_path))
    if source is None:
        raise FileNotFoundError(image_path)
    height, width = source.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic_resistor_sequence.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        15.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("não foi possível criar o vídeo sintético")
    matrices: list[np.ndarray] = []
    for frame_index in range(frames):
        angle = 4.0 * math.sin(frame_index / 6.0)
        dx = 5.0 * math.sin(frame_index / 8.0)
        dy = 3.0 * math.cos(frame_index / 7.0)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[:, 2] += (dx, dy)
        transformed = cv2.warpAffine(source, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        brightness = 1.0 + 0.08 * math.sin(frame_index / 5.0)
        transformed = np.clip(transformed.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        writer.write(transformed)
        matrices.append(matrix)
    writer.release()

    detector = YOLO(str(weights))
    # Exclude model initialization from the per-frame latency statistics.
    detector.predict(
        source=source,
        imgsz=imgsz,
        conf=confidence,
        device=device,
        verbose=False,
        save=False,
    )
    tracker = IoUTracker(iou_threshold=0.25, max_missed=2)
    track_history: dict[int, list[int]] = {}
    ground_truth: list[SampleRecord] = []
    predictions: list[SampleRecord] = []
    capture = cv2.VideoCapture(str(video_path))
    started = time.perf_counter()
    frame_index = 0
    while frame_index < frames:
        ok, frame = capture.read()
        if not ok:
            break
        sample_id = f"{seed.sample_id}-video-{frame_index:04d}"
        expected_objects = tuple(
            ObjectRecord(
                bbox=_transform_bbox(obj.bbox, matrices[frame_index], (height, width)),
                bands=obj.bands,
                ohms=obj.ohms,
                tolerance_percent=obj.tolerance_percent,
            )
            for obj in seed.objects
        )
        ground_truth.append(
            SampleRecord(
                sample_id=sample_id,
                source_id=seed.source_id or seed.sample_id,
                frame=frame_index,
                objects=expected_objects,
                media_type="video",
            )
        )
        inference_started = time.perf_counter()
        result = detector.predict(
            source=frame,
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
                class_name = names[int(class_id)] if isinstance(names, (list, tuple)) else names[int(class_id)]
                if str(class_name).lower() != "resistor":
                    continue
                left, top, right, bottom = box
                boxes.append((left, top, right - left, bottom - top))
                scores.append(float(score))
        track_ids = tracker.update(boxes)
        for track_id in track_ids:
            track_history.setdefault(track_id, []).append(frame_index)
        predicted_objects = tuple(
            ObjectRecord(bbox=bbox, confidence=score, track_id=track_id)
            for bbox, score, track_id in zip(boxes, scores, track_ids)
        )
        predictions.append(
            SampleRecord(
                sample_id=sample_id,
                source_id=seed.source_id or seed.sample_id,
                frame=frame_index,
                objects=predicted_objects,
                media_type="video",
                latency_ms=(time.perf_counter() - inference_started) * 1000,
                strategy="video-yolo-tracker",
            )
        )
        frame_index += 1
    capture.release()
    metrics = evaluate(ground_truth, predictions, strategy="video-yolo-tracker")
    summary = {
        "weights": str(weights),
        "seed_image": str(image_path),
        "video": str(video_path),
        "frames_requested": frames,
        "frames_evaluated": len(predictions),
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "tracking": {
            "unique_tracks": len(track_history),
            "detections_with_track": sum(len(frames) for frames in track_history.values()),
            "mean_track_length": (
                statistics.mean(len(frames) for frames in track_history.values())
                if track_history
                else 0.0
            ),
            "max_track_length": max(
                (len(frames) for frames in track_history.values()), default=0
            ),
        },
        "benchmark": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    summary = run(
        args.weights,
        args.manifest,
        args.output_dir,
        frames=args.frames,
        imgsz=args.imgsz,
        confidence=args.confidence,
        device=args.device,
    )
    benchmark = summary.pop("benchmark")
    print(
        json.dumps(
            {
                **{key: value for key, value in summary.items() if key != "benchmark"},
                "f1": benchmark["objects"]["f1"],
                "precision": benchmark["objects"]["precision"],
                "recall": benchmark["objects"]["recall"],
                "fps": benchmark["performance"]["fps_estimate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
