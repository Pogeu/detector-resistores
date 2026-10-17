"""Headless port of Automatic Component Analyzer's static-frame path.

The upstream project is an interactive USB-camera application.  This module
keeps its default thresholds, contour ordering, band rejection, color rules,
orientation logic, and 4-band decoder, but exposes them for deterministic
image/video benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DIGIT_MAP = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "silver": 8,
    "white": 9,
}

MULTIPLIER_MAP = {
    "black": 1,
    "brown": 10,
    "red": 100,
    "orange": 1_000,
    "yellow": 10_000,
    "green": 100_000,
    "blue": 1_000_000,
    "violet": 10_000_000,
    "gold": 0.1,
    "silver": 0.01,
}

TOLERANCE_MAP = {
    "brown": "1%",
    "red": "2%",
    "green": "0.5%",
    "blue": "0.25%",
    "violet": "0.1%",
    "gray": "0.05%",
    "gold": "5%",
    "silver": "10%",
}


@dataclass(frozen=True)
class AnalyzerConfig:
    """The default trackbar values in upstream ``main.py``."""

    gaussian_kernel_radius: int = 15
    dark_threshold: int = 40


def decode_4_band(bands: list[dict[str, Any]]) -> tuple[float, str] | None:
    """Match upstream ``decode_4_band`` exactly."""

    if len(bands) != 4:
        return None
    try:
        d1 = DIGIT_MAP[bands[0]["color"]]
        d2 = DIGIT_MAP[bands[1]["color"]]
        multiplier = MULTIPLIER_MAP[bands[2]["color"]]
        tolerance = TOLERANCE_MAP.get(bands[3]["color"], "unknown")
        return (d1 * 10 + d2) * multiplier, tolerance
    except KeyError:
        return None


def _classify_color(mean_h: float, mean_s: float, mean_v: float) -> str:
    """Match the upstream HSV decision tree, including its thresholds."""

    if mean_v < 20:
        return "black"
    if mean_s < 20:
        return "white" if mean_v > 170 else "silver"
    if mean_h < 8 or mean_h > 170:
        return "red"
    if mean_h < 24:
        if mean_h < 16:
            if mean_v > 140:
                return "orange"
            if mean_v < 80:
                return "gold"
            return "brown"
        return "orange" if mean_v > 140 else "gold"
    if mean_h < 40:
        return "yellow"
    if mean_h < 94:
        return "green"
    if mean_h < 118:
        return "blue"
    return "violet"


def analyze_frame(
    image_bgr: np.ndarray,
    config: AnalyzerConfig | None = None,
) -> dict[str, Any]:
    """Analyze one BGR image using the upstream static-frame algorithm.

    The returned structure is JSON-serializable and intentionally contains
    enough intermediate information to audit a benchmark disagreement.
    """

    config = config or AnalyzerConfig()
    radius = int(config.gaussian_kernel_radius)
    kernel_size = radius * 2 + 1
    blur = cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    light_colors_mask = cv2.inRange(
        hsv, np.array([0, 140, 0]), np.array([179, 255, 255])
    )
    white_mask = cv2.inRange(
        hsv, np.array([30, 0, 170]), np.array([65, 255, 255])
    )
    silver_mask = cv2.inRange(
        hsv, np.array([35, 0, 130]), np.array([85, 20, 160])
    )
    light_colors_mask = cv2.bitwise_or(light_colors_mask, white_mask)
    light_colors_mask = cv2.bitwise_or(light_colors_mask, silver_mask)
    _, dark_colors_mask = cv2.threshold(
        gray, int(config.dark_threshold), 255, cv2.THRESH_BINARY_INV
    )
    all_colors_mask = cv2.bitwise_or(light_colors_mask, dark_colors_mask)

    all_contours, _ = cv2.findContours(
        all_colors_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_sorted = sorted(all_contours, key=cv2.contourArea, reverse=True)
    bands: list[dict[str, Any]] = []
    rejected = {"area": 0, "aspect_ratio_break": 0, "duplicate_center": 0}

    for contour in contours_sorted:
        if len(bands) >= 4:
            break
        area = float(cv2.contourArea(contour))
        if area < 100:
            rejected["area"] += 1
            continue
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        aspect_ratio = w / h if h else float("inf")
        # This is deliberately a break, not continue: it is what main.py does.
        if aspect_ratio > 2:
            rejected["aspect_ratio_break"] += 1
            break
        if any(abs(center_x - band["center_x"]) < 40 for band in bands):
            rejected["duplicate_center"] += 1
            continue

        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        roi = blur[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]
        if roi.size == 0:
            continue
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = (
            float(np.mean(roi_hsv[:, :, channel])) for channel in range(3)
        )
        bands.append(
            {
                "center_x": int(center_x),
                "center_y": int(center_y),
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": area,
                "aspect_ratio": float(aspect_ratio),
                "mean_h": mean_h,
                "mean_s": mean_s,
                "mean_v": mean_v,
                "color": _classify_color(mean_h, mean_s, mean_v),
            }
        )

    bands.sort(key=lambda band: band["center_x"])
    colors = [band["color"] for band in bands] if len(bands) == 4 else []
    left_is_gold = bool(colors and colors[0] == "gold")
    right_is_gold = bool(colors and colors[-1] == "gold")
    decoded_results: list[dict[str, Any]] = []

    if right_is_gold:
        reading = bands
    elif left_is_gold:
        reading = list(reversed(bands))
    else:
        reading = None

    readings = [reading] if reading else [bands, list(reversed(bands))]
    for oriented_bands in readings:
        decoded = decode_4_band(oriented_bands)
        if decoded:
            value, tolerance = decoded
            decoded_results.append(
                {"value": float(value), "tolerance": tolerance}
            )

    return {
        "band_count": len(bands),
        "colors": colors,
        "bands": bands,
        "decoded_results": decoded_results,
        "rejected": rejected,
        "config": {
            "gaussian_kernel_radius": radius,
            "dark_threshold": int(config.dark_threshold),
        },
    }

