"""Train and validate a YOLO segmentation model on a prepared dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(data_yaml: Path, output_dir: Path, model_name: str, epochs: int, imgsz: int, batch: int, device: str) -> dict:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        workers=2,
        patience=15,
        plots=True,
        verbose=False,
    )
    best = output_dir / "train" / "weights" / "best.pt"
    metrics = model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, device=device, plots=True)
    summary = {
        "model": model_name,
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "best_weights": str(best),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
        "mask_precision": float(metrics.seg.p[0]) if len(metrics.seg.p) else 0.0,
        "mask_recall": float(metrics.seg.r[0]) if len(metrics.seg.r) else 0.0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    print(json.dumps(train(args.data_yaml, args.output_dir, args.model, args.epochs, args.imgsz, args.batch, args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

