from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detection import analyze_frame, annotate_frame
from .pipeline import iter_video_frames, read_image


def _detection_to_dict(detection) -> dict:
    value = detection.value
    return {
        "bbox": {
            "x": detection.bbox[0],
            "y": detection.bbox[1],
            "width": detection.bbox[2],
            "height": detection.bbox[3],
        },
        "bands": list(detection.bands),
        "value_ohms": value.ohms if value else None,
        "value": value.formatted if value else None,
        "tolerance_percent": value.tolerance_percent if value else None,
        "confidence": round(detection.confidence, 3),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conta e identifica resistores em imagens e vídeos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("image", help="analisa uma imagem")
    image.add_argument("input", type=Path)
    image.add_argument("--band-count", type=int, choices=(4, 5), default=4)
    image.add_argument("--output", type=Path, help="salva a imagem anotada")

    video = subparsers.add_parser("video", help="analisa um vídeo")
    video.add_argument("input", type=Path)
    video.add_argument("--band-count", type=int, choices=(4, 5), default=4)
    video.add_argument("--every-n-frames", type=int, default=1)
    video.add_argument("--max-frames", type=int)
    video.add_argument("--output", type=Path, help="salva um vídeo anotado")
    return parser


def _run_image(args: argparse.Namespace) -> None:
    image = read_image(args.input)
    detections = analyze_frame(image, band_count=args.band_count)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), annotate_frame(image, detections))
    print(json.dumps({"input": str(args.input), "count": len(detections), "detections": [_detection_to_dict(item) for item in detections]}, ensure_ascii=False, indent=2))


def _run_video(args: argparse.Namespace) -> None:
    writer = None
    frames = []
    try:
        for frame_index, frame, detections in iter_video_frames(
            args.input,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_frames,
            band_count=args.band_count,
        ):
            annotated = annotate_frame(frame, detections)
            if args.output and writer is None:
                height, width = frame.shape[:2]
                capture = cv2.VideoCapture(str(args.input))
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                capture.release()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / max(args.every_n_frames, 1),
                    (width, height),
                )
            if writer is not None:
                writer.write(annotated)
            frames.append({"frame": frame_index, "count": len(detections), "detections": [_detection_to_dict(item) for item in detections]})
    finally:
        if writer is not None:
            writer.release()
    print(json.dumps({"input": str(args.input), "frames": frames}, ensure_ascii=False, indent=2))


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "image":
        _run_image(args)
    else:
        _run_video(args)

