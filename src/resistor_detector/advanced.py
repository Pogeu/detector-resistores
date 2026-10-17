"""Variantes clássicas adicionais para o benchmark."""

from __future__ import annotations

from typing import Callable, Iterable

import cv2
import numpy as np

from .colors import classify_bgr, decode_color_bands
from .detection import Candidate, _intersection_over_union, sample_band_colors
from .models import ResistorDetection


ColorClassifier = Callable[[Iterable[int]], str | None]


def segment_resistor_mask(
    frame: np.ndarray,
    *,
    saturation_min: int = 35,
    use_contrast: bool = True,
) -> np.ndarray:
    """Gera máscara de foreground usando contraste local e saturação.

    É uma segmentação por regras, não uma máscara anotada de aprendizado.
    Serve para comparar a etapa de isolamento antes do detector de contornos.
    """

    if frame is None or frame.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    local_background = cv2.medianBlur(gray, 31)
    contrast = cv2.absdiff(gray, local_background)
    _, contrast_mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_mask = cv2.inRange(hsv[:, :, 1], saturation_min, 255)
    mask = (
        cv2.bitwise_or(contrast_mask, saturation_mask)
        if use_contrast
        else saturation_mask
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_segmented_candidates(
    frame: np.ndarray,
    *,
    min_area: float = 350.0,
) -> list[Candidate]:
    """Extrai candidatos alongados da máscara da segmentação por regras."""

    mask = segment_resistor_mask(frame)
    if mask.size == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    candidates: list[Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > frame_area * 0.8:
            continue
        _, (width, height), _ = cv2.minAreaRect(contour)
        short_side, long_side = sorted((width, height))
        if short_side < 8 or long_side / max(short_side, 1.0) < 1.8 or long_side / max(short_side, 1.0) > 12:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 20 or height < 10:
            continue
        candidates.append(Candidate((x, y, width, height), area))
    candidates.sort(key=lambda item: item.area, reverse=True)
    selected: list[Candidate] = []
    for candidate in candidates:
        if all(_intersection_over_union(candidate, other) < 0.45 for other in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item.bbox[1], item.bbox[0]))


def _decode_candidate(
    frame: np.ndarray,
    candidate: Candidate,
    band_count: int,
    classifier: ColorClassifier,
) -> ResistorDetection:
    bands = sample_band_colors(
        frame,
        candidate.bbox,
        band_count=band_count,
        classifier=classifier,
    )
    value = decode_color_bands(bands)
    confidence = sum(color is not None for color in bands) / band_count
    return ResistorDetection(
        bbox=candidate.bbox,
        bands=tuple(color or "unknown" for color in bands),
        value=value,
        confidence=confidence,
    )


def analyze_segmented_frame(
    frame: np.ndarray,
    *,
    band_count: int = 4,
    min_area: float = 350.0,
    classifier: ColorClassifier = classify_bgr,
) -> list[ResistorDetection]:
    """Executa segmentação por regras seguida do decoder de faixas."""

    return [
        _decode_candidate(frame, candidate, band_count, classifier)
        for candidate in find_segmented_candidates(frame, min_area=min_area)
    ]


def analyze_frame_auto(
    frame: np.ndarray,
    *,
    min_area: float = 350.0,
    classifier: ColorClassifier = classify_bgr,
) -> list[ResistorDetection]:
    """Escolhe 4 ou 5 faixas por confiança de reconhecimento/decodificação."""

    from .detection import find_candidates

    detections: list[ResistorDetection] = []
    for candidate in find_candidates(frame, min_area=min_area):
        options = [_decode_candidate(frame, candidate, count, classifier) for count in (4, 5)]
        detections.append(
            max(
                options,
                key=lambda item: (
                    1 if item.value is not None else 0,
                    sum(color != "unknown" for color in item.bands),
                    item.confidence,
                ),
            )
        )
    return detections
