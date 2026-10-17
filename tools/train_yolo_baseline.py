"""Treina e avalia um detector YOLO no formato do benchmark local."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

# Ultralytics 8.3.x still calls np.trapz; NumPy 2.x renamed it to
# np.trapezoid. Keep the benchmark reproducible without changing the user's
# environment.
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from resistor_detector.benchmark import ObjectRecord, SampleRecord, evaluate, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _resolved_data_yaml(data_yaml: Path, output_dir: Path) -> Path:
    """Materializa um YAML sem depender do datasets_dir global do Ultralytics."""

    import yaml

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    config: dict[str, Any] = {"names": raw["names"], "nc": raw.get("nc", len(raw["names"]))}
    for split in ("train", "val", "test"):
        value = raw.get(split)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (data_yaml.parent / path).resolve()
            if not path.exists() and value.startswith("../"):
                # Alguns exports Universe escrevem ../train/images embora
                # o pacote extraído contenha extracted/train/images.
                alternative = (data_yaml.parent / value[3:]).resolve()
                if alternative.exists():
                    path = alternative
        config[split] = path.as_posix()
    # Alguns exports do Roboflow fornecem apenas train/valid. O benchmark
    # ainda pode avaliar o split test do manifesto; para a métrica nativa do
    # Ultralytics usamos valid como fallback explícito.
    if "test" not in config and "val" in config:
        config["test"] = config["val"]
    output_path = output_dir / "resolved-data.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def _load_split(path: Path, split: str) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw.get("split") == split:
            records.append(SampleRecord.from_mapping(raw))
    return records


def train_and_evaluate(
    data_yaml: Path,
    ground_truth_path: Path,
    output_dir: Path,
    *,
    model_name: str = "yolo11n.pt",
    epochs: int = 5,
    imgsz: int = 416,
    batch: int = 8,
    confidence: float = 0.25,
    device: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("instale ultralytics para treinar o detector YOLO") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = "0" if device == "auto" and torch.cuda.is_available() else device
    resolved_data_yaml = _resolved_data_yaml(data_yaml, output_dir)
    model = YOLO(model_name)
    started = time.perf_counter()
    model.train(
        data=str(resolved_data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        workers=0,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=False,
        plots=False,
    )
    best_path = Path(model.trainer.best)
    trained_model = YOLO(str(best_path))
    validation = trained_model.val(
        data=str(resolved_data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        workers=0,
        plots=False,
        verbose=False,
    )
    ground_truth = _load_split(ground_truth_path, "test")
    predictions: list[SampleRecord] = []
    for sample in ground_truth:
        if not sample.path:
            predictions.append(SampleRecord(sample_id=sample.sample_id, objects=()))
            continue
        image_path = ROOT / sample.path
        inference_started = time.perf_counter()
        result = trained_model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=confidence,
            device=resolved_device,
            verbose=False,
            save=False,
        )[0]
        objects: list[ObjectRecord] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()
            names = trained_model.names
            for box, score, class_id in zip(boxes, confidences, class_ids):
                class_name = names[int(class_id)] if isinstance(names, (list, tuple)) else names[int(class_id)]
                if str(class_name).strip().lower() != "resistor":
                    continue
                left, top, right, bottom = box
                objects.append(
                    ObjectRecord(
                        bbox=(left, top, right - left, bottom - top),
                        confidence=float(score),
                    )
                )
        predictions.append(
            SampleRecord(
                sample_id=sample.sample_id,
                source_id=sample.source_id,
                objects=tuple(objects),
                latency_ms=(time.perf_counter() - inference_started) * 1000,
                strategy="detector-only",
            )
        )

    prediction_path = output_dir / "test-predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    benchmark_metrics = evaluate(ground_truth, predictions, strategy="detector-only")
    box = validation.box
    summary = {
        "model": model_name,
        "best_checkpoint": str(best_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "test_samples": len(ground_truth),
        "elapsed_seconds": time.perf_counter() - started,
        "target_class": "Resistor",
        "ultralytics": {
            "device": resolved_device,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "benchmark": benchmark_metrics,
        "predictions": str(prediction_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu ou índice CUDA (ex.: 0)")
    args = parser.parse_args(argv)
    summary = train_and_evaluate(
        args.data_yaml,
        args.ground_truth,
        args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        confidence=args.confidence,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "benchmark"}, indent=2))
    print(json.dumps(summary["benchmark"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
