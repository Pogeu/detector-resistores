"""Port of the MATLAB File Exchange resistor pipeline.

This module intentionally keeps the original thresholds and heuristics. It is
used for a fair MATLAB-versus-Python comparison, not as the production
detector. The production code may use more robust segmentation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


COLOR_DIGITS = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "grey": 8,
    "white": 9,
}

# MATLAB rgb2hsv uses [0, 1] for all three channels. The red interval wraps
# around zero, exactly as createMaskred.m does.
COLOR_THRESHOLDS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "black": ((0.045, 0.115), (0.243, 0.944), (0.071, 0.173)),
    "brown": ((0.011, 0.090), (0.398, 0.831), (0.179, 0.328)),
    "green": ((0.265, 0.331), (0.403, 0.636), (0.302, 0.469)),
    "orange": ((0.054, 0.081), (0.508, 0.892), (0.484, 0.778)),
    "red": ((0.985, 0.024), (0.473, 0.871), (0.351, 0.642)),
    "red_alt": ((0.023, 0.036), (0.642, 0.892), (0.471, 0.584)),
    "violet": ((0.939, 0.996), (0.274, 0.425), (0.296, 0.452)),
    "yellow": ((0.125, 0.139), (0.669, 0.825), (0.632, 0.826)),
    "yellow_alt": ((0.117, 0.126), (0.593, 0.678), (0.788, 0.945)),
    "yellow_alt2": ((0.121, 0.134), (0.492, 0.619), (0.643, 0.792)),
}


@dataclass(frozen=True)
class MatlabCompatDetection:
    bbox: tuple[int, int, int, int]
    bands: tuple[str, ...]
    value_ohms: int | None


def _hsv01(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] /= 360.0
    return hsv


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    return ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)


def _morph(mask: np.ndarray, operation: int, kernel: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    flood = binary.copy()
    flood_mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def _component_stats(mask: np.ndarray) -> list[dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=4
    )
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        area = int(stats[label, cv2.CC_STAT_AREA])
        if len(xs) < 2:
            eccentricity = 0.0
        else:
            coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
            covariance = np.cov(coords, rowvar=False, bias=True)
            eigenvalues = np.linalg.eigvalsh(covariance)
            major = float(max(eigenvalues))
            minor = float(min(eigenvalues))
            eccentricity = 0.0 if major <= 1e-12 else float(np.sqrt(max(0.0, 1.0 - minor / major)))
        components.append(
            {
                "area": area,
                "eccentricity": eccentricity,
                "bbox": (
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ),
            }
        )
    return components


def _threshold_hsv(hsv: np.ndarray, thresholds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    (h_min, h_max), (s_min, s_max), (v_min, v_max) = thresholds
    if h_min <= h_max:
        hue = (hsv[..., 0] >= h_min) & (hsv[..., 0] <= h_max)
    else:
        hue = (hsv[..., 0] >= h_min) | (hsv[..., 0] <= h_max)
    return (hue & (hsv[..., 1] >= s_min) & (hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min) & (hsv[..., 2] <= v_max)).astype(np.uint8) * 255


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(image.shape[1], int(np.ceil(x + width)))
    y1 = min(image.shape[0], int(np.ceil(y + height)))
    return image[y0:y1, x0:x1]


def _main_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    binary = _morph(binary, cv2.MORPH_CLOSE, _disk(12))
    binary = _fill_holes(binary)
    hsv = _hsv01(image_bgr)
    average_v = float(np.mean(hsv[..., 2]))
    side = 41 if average_v < 0.9 else 18
    opened = _morph(binary, cv2.MORPH_OPEN, np.ones((side, side), dtype=np.uint8))
    return opened, average_v


def _accept_resistor(crop: np.ndarray) -> bool:
    hsv = _hsv01(crop)
    first = _threshold_hsv(hsv, ((0.114, 0.146), (0.295, 0.440), (0.510, 0.695)))
    if _component_stats(_morph(first, cv2.MORPH_OPEN, _disk(6))):
        return True
    second = _threshold_hsv(hsv, ((0.101, 0.148), (0.240, 0.470), (0.715, 0.970)))
    return bool(_component_stats(_morph(second, cv2.MORPH_OPEN, _disk(2))))


def _largest_band(crop: np.ndarray, hsv: np.ndarray, name: str, radius: int) -> dict[str, Any] | None:
    mask = _threshold_hsv(hsv, COLOR_THRESHOLDS[name])
    components = _component_stats(_morph(mask, cv2.MORPH_OPEN, _disk(radius)))
    if not components:
        return None
    selected = max(components, key=lambda item: item["area"])
    return {"name": name, **selected}


def _find_bands(crop: np.ndarray, average_v: float) -> tuple[str, ...]:
    hsv = _hsv01(crop)
    bands: list[dict[str, Any]] = []
    sequence = [
        ("brown", 1),
        ("orange", 1),
        ("red" if average_v < 0.9 else "red_alt", 0),
        ("black", 0),
    ]
    for name, radius in sequence:
        band = _largest_band(crop, hsv, name, radius)
        if band is not None:
            bands.append(band)

    for name in ("yellow_alt", "yellow_alt2", "yellow"):
        band = _largest_band(crop, hsv, name, 0)
        if band is not None:
            bands.append(band)
            break

    for name in ("violet", "green"):
        band = _largest_band(crop, hsv, name, 0)
        if band is not None:
            bands.append(band)

    if not bands:
        return ()
    if len(bands) < 3:
        largest = max(bands, key=lambda item: item["area"])
        while len(bands) < 3:
            bands.append(dict(largest))

    horizontal = crop.shape[1] >= crop.shape[0]
    coordinate = 0 if horizontal else 1
    positions = [int(item["bbox"][coordinate]) for item in bands[:3]]
    if sum(positions) < 250:
        first_index = int(np.argmin(positions))
        third_index = int(np.argmax(positions))
    else:
        first_index = int(np.argmax(positions))
        third_index = int(np.argmin(positions))
    middle_index = next(index for index in range(3) if index not in (first_index, third_index))
    ordered = [bands[first_index], bands[middle_index], bands[third_index]]
    return tuple(item["name"].replace("_alt2", "").replace("_alt", "") for item in ordered)


def analyze_matlab_compatible(image_bgr: np.ndarray) -> dict[str, Any]:
    """Run the MATLAB package's detection logic as closely as OpenCV allows."""

    main_mask, average_v = _main_mask(image_bgr)
    components = _component_stats(main_mask)
    candidates = [item for item in components if item["eccentricity"] > 0.0]
    detections: list[MatlabCompatDetection] = []
    for candidate in candidates:
        crop = _crop(image_bgr, candidate["bbox"])
        if not crop.size or not _accept_resistor(crop):
            continue
        bands = _find_bands(crop, average_v)
        value = None
        if len(bands) >= 3 and all(band in COLOR_DIGITS for band in bands[:3]):
            value = int((COLOR_DIGITS[bands[0]] * 10 + COLOR_DIGITS[bands[1]]) * 10 ** COLOR_DIGITS[bands[2]])
        detections.append(MatlabCompatDetection(candidate["bbox"], bands, value))
    return {
        "candidate_count": len(candidates),
        "accepted_count": len(detections),
        "detections": [
            {"bbox": list(item.bbox), "bands": list(item.bands), "value_ohms": item.value_ohms}
            for item in detections
        ],
        "values": [item.value_ohms for item in detections],
    }
