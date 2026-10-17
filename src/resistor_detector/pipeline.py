from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from .detection import analyze_frame
from .tracking import IoUTracker


def read_image(path: str | Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {path}")
    return image


def iter_video_frames(
    path: str | Path,
    *,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    band_count: int = 4,
) -> Iterator[tuple[int, object, list]]:
    """Produz (índice, quadro, detecções) para os quadros selecionados."""

    if every_n_frames < 1:
        raise ValueError("every_n_frames deve ser maior ou igual a 1")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")

    frame_index = 0
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n_frames == 0:
                yield frame_index, frame, analyze_frame(frame, band_count=band_count)
                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
            frame_index += 1
    finally:
            capture.release()


def iter_tracked_video_frames(
    path: str | Path,
    *,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    band_count: int = 4,
    iou_threshold: float = 0.25,
) -> Iterator[tuple[int, object, list, list[str]]]:
    """Produz quadros com detecções e IDs estáveis do tracker IoU."""

    tracker = IoUTracker(iou_threshold=iou_threshold)
    for frame_index, frame, detections in iter_video_frames(
        path,
        every_n_frames=every_n_frames,
        max_frames=max_frames,
        band_count=band_count,
    ):
        track_ids = tracker.update([detection.bbox for detection in detections])
        yield frame_index, frame, detections, track_ids
