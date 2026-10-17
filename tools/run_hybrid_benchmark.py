"""Executa o híbrido detector YOLO + leitura clássica ou CNN."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from resistor_detector.benchmark import (
    ObjectRecord,
    SampleRecord,
    evaluate,
    write_jsonl,
)
from resistor_detector.caoile_cnn import build_model
from resistor_detector.colors import decode_color_bands
from resistor_detector.detection import sample_band_colors


ROOT = Path(__file__).resolve().parents[1]


def _load_manifest(path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            records.append(SampleRecord.from_mapping(json.loads(line)))
    return records


def _load_cnn(checkpoint_path: Path, device: torch.device) -> tuple[Any, Any, dict[int, str], dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model"]
    model = build_model(
        len(checkpoint["class_to_index"]),
        int(model_config["first_filters"]),
        int(model_config["second_filters"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    protocol = checkpoint.get("protocol", {})
    operations: list[Any] = []
    if protocol.get("center_crop_size") is not None:
        operations.append(transforms.CenterCrop(int(protocol["center_crop_size"])))
    operations.extend([transforms.Resize((64, 64)), transforms.ToTensor()])
    inverse = {int(index): token for token, index in checkpoint["class_to_index"].items()}
    return model, transforms.Compose(operations), inverse, checkpoint["class_values_ohms"]


def _predict_cnn(
    model: Any,
    transform: Any,
    inverse: dict[int, str],
    class_values_ohms: dict[str, float],
    crop: Any,
    device: torch.device,
) -> tuple[str, float, float]:
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    token = inverse[index]
    return token, float(class_values_ohms[token]), float(probabilities[index])


def run(
    weights: Path,
    ground_truth_path: Path,
    output_dir: Path,
    *,
    cnn_checkpoint: Path | None = None,
    decoder: str = "classical",
    band_count: int = 4,
    imgsz: int = 640,
    confidence: float = 0.1,
    device_name: str = "0",
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name != "cpu" and torch.cuda.is_available() else "cpu")
    detector = YOLO(str(weights))
    cnn = None
    if decoder == "cnn":
        if cnn_checkpoint is None:
            raise ValueError("--cnn-checkpoint é obrigatório com --decoder cnn")
        cnn = _load_cnn(cnn_checkpoint, device)

    ground_truth = _load_manifest(ground_truth_path)
    predictions: list[SampleRecord] = []
    started = time.perf_counter()
    for sample in ground_truth:
        image_path = Path(sample.path or "")
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(image_path)
        inference_started = time.perf_counter()
        result = detector.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=confidence,
            device=device_name,
            verbose=False,
            save=False,
        )[0]
        objects: list[ObjectRecord] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            names = detector.names
            for box, score, class_id in zip(boxes, scores, classes):
                class_name = names[int(class_id)] if isinstance(names, (list, tuple)) else names[int(class_id)]
                if str(class_name).strip().lower() != "resistor":
                    continue
                left, top, right, bottom = box
                x, y = max(0, int(left)), max(0, int(top))
                width, height = max(1, int(right - left)), max(1, int(bottom - top))
                crop = frame[y : y + height, x : x + width]
                bands = tuple(
                    color for color in sample_band_colors(frame, (x, y, width, height), band_count)
                    if color is not None
                )
                value = decode_color_bands(bands)
                ohms = value.ohms if value is not None else None
                tolerance = value.tolerance_percent if value is not None else None
                if decoder == "cnn" and cnn is not None and crop.size:
                    _, ohms, cnn_confidence = _predict_cnn(*cnn, crop, device)
                    score = float(score) * cnn_confidence
                objects.append(
                    ObjectRecord(
                        bbox=(left, top, right - left, bottom - top),
                        bands=bands,
                        ohms=ohms,
                        tolerance_percent=tolerance,
                        confidence=float(score),
                    )
                )
        predictions.append(
            SampleRecord(
                sample_id=sample.sample_id,
                source_id=sample.source_id,
                path=sample.path,
                objects=tuple(objects),
                latency_ms=(time.perf_counter() - inference_started) * 1000,
                strategy=f"hybrid-yolo-{decoder}-{band_count}",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    metrics = evaluate(ground_truth, predictions, strategy=f"hybrid-yolo-{decoder}-{band_count}")
    summary = {
        "weights": str(weights),
        "cnn_checkpoint": None if cnn_checkpoint is None else str(cnn_checkpoint),
        "decoder": decoder,
        "band_count": band_count,
        "device": str(device),
        "samples": len(ground_truth),
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": str(prediction_path),
        "benchmark": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cnn-checkpoint", type=Path)
    parser.add_argument("--decoder", choices=("classical", "cnn"), default="classical")
    parser.add_argument("--band-count", type=int, choices=(4, 5), default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    summary = run(
        args.weights,
        args.ground_truth,
        args.output_dir,
        cnn_checkpoint=args.cnn_checkpoint,
        decoder=args.decoder,
        band_count=args.band_count,
        imgsz=args.imgsz,
        confidence=args.confidence,
        device_name=args.device,
    )
    benchmark = summary.pop("benchmark")
    print(json.dumps(summary, indent=2))
    print(json.dumps({key: benchmark[key] for key in ("objects", "bands", "value", "performance")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
