from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from .colors import classify_bgr, decode_color_bands
from .models import ResistorDetection


@dataclass(frozen=True)
class Candidate:
    bbox: tuple[int, int, int, int]
    area: float


def _intersection_over_union(left: Candidate, right: Candidate) -> float:
    lx, ly, lw, lh = left.bbox
    rx, ry, rw, rh = right.bbox
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def find_candidates(frame: np.ndarray, min_area: float = 350.0) -> list[Candidate]:
    """Encontra formas alongadas que podem ser corpos de resistores."""

    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > frame_area * 0.8:
            continue

        _, (width, height), _ = cv2.minAreaRect(contour)
        short_side, long_side = sorted((width, height))
        if short_side < 8 or long_side / short_side < 1.8 or long_side / short_side > 12:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 10:
            continue
        candidates.append(Candidate((x, y, w, h), area))

    candidates.sort(key=lambda item: item.area, reverse=True)
    selected: list[Candidate] = []
    for candidate in candidates:
        if all(_intersection_over_union(candidate, other) < 0.45 for other in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item.bbox[1], item.bbox[0]))


def sample_band_colors(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    band_count: int = 4,
    classifier: Callable[[Iterable[int]], str | None] = classify_bgr,
) -> tuple[str | None, ...]:
    """Amostra o centro de faixas igualmente espaçadas no corpo candidato."""

    x, y, width, height = bbox
    crop = frame[max(0, y) : y + height, max(0, x) : x + width]
    if crop.size == 0:
        return tuple(None for _ in range(band_count))
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    crop = cv2.resize(crop, (240, 80), interpolation=cv2.INTER_AREA)
    strip = crop[22:58, :]
    colors: list[str | None] = []
    for position in np.linspace(0.18, 0.82, band_count):
        center = int(position * (strip.shape[1] - 1))
        start, end = max(0, center - 7), min(strip.shape[1], center + 8)
        median_bgr = np.median(strip[:, start:end], axis=(0, 1)).astype(np.uint8)
        colors.append(classifier(median_bgr))
    return tuple(colors)


def analyze_frame(
    frame: np.ndarray,
    *,
    band_count: int = 4,
    min_area: float = 350.0,
    classifier: Callable[[Iterable[int]], str | None] = classify_bgr,
) -> list[ResistorDetection]:
    """Detecta e tenta identificar todos os resistores de um quadro."""

    if band_count not in (4, 5):
        raise ValueError("band_count deve ser 4 ou 5")

    detections: list[ResistorDetection] = []
    for candidate in find_candidates(frame, min_area=min_area):
        bands = sample_band_colors(
            frame,
            candidate.bbox,
            band_count=band_count,
            classifier=classifier,
        )
        value = decode_color_bands(bands)
        recognized = sum(color is not None for color in bands)
        confidence = recognized / band_count
        detections.append(
            ResistorDetection(
                bbox=candidate.bbox,
                bands=tuple(color or "unknown" for color in bands),
                value=value,
                confidence=confidence,
            )
        )
    return detections


def annotate_frame(frame: np.ndarray, detections: list[ResistorDetection]) -> np.ndarray:
    """Desenha caixas e valores sobre uma cópia do quadro."""

    annotated = frame.copy()
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection.bbox
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 220, 0), 2)
        text = f"#{index} {detection.label}"
        cv2.putText(
            annotated,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated
