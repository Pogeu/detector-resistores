from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .models import ResistanceValue


DIGITS: dict[str, int] = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "gray": 8,
    "white": 9,
}

MULTIPLIERS: dict[str, int] = {**DIGITS, "gold": -1, "silver": -2}

TOLERANCES: dict[str, float] = {
    "brown": 1.0,
    "red": 2.0,
    "green": 0.5,
    "blue": 0.25,
    "violet": 0.1,
    "gray": 0.05,
    "gold": 5.0,
    "silver": 10.0,
}


def classify_bgr(bgr: Iterable[int]) -> str | None:
    """Classifica uma cor BGR usando regras simples no espaço HSV.

    A classificação é deliberadamente conservadora: pixels pouco saturados ou
    fora das faixas conhecidas retornam ``None``.
    """

    pixel = np.uint8([[list(bgr)]])
    hue, saturation, value = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hue), int(saturation), int(value)

    if v < 55:
        return "black"
    if s < 35:
        if v >= 205:
            return "white"
        if v >= 115:
            return "silver"
        return "gray"

    if (h <= 5 or h >= 170) and s >= 70:
        return "red"
    if 5 < h < 18:
        if v < 145:
            return "brown"
        if s < 175 and v > 145:
            return "gold"
        return "orange"
    if 18 <= h < 38:
        return "yellow"
    if 38 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    if 130 <= h < 170:
        return "violet"
    return None


# Protótipos aproximados em BGR. A projeção PCA reduz Lab a duas dimensões e
# deixa a decisão menos dependente de hue circular, especialmente em sombras.
_LAB_PROTOTYPE_NAMES = (
    "black",
    "brown",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "violet",
    "gray",
    "white",
    "silver",
    "gold",
)
_LAB_PROTOTYPE_BGR = np.uint8(
    [
        (0, 0, 0),
        (42, 42, 80),
        (0, 0, 190),
        (0, 120, 235),
        (0, 200, 220),
        (0, 150, 0),
        (180, 80, 0),
        (150, 50, 150),
        (128, 128, 128),
        (235, 235, 235),
        (185, 185, 185),
        (20, 170, 210),
    ]
)
_LAB_PROTOTYPES = cv2.cvtColor(_LAB_PROTOTYPE_BGR.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
_LAB_MEAN = _LAB_PROTOTYPES.mean(axis=0)
_, _, _LAB_COMPONENTS = np.linalg.svd(_LAB_PROTOTYPES - _LAB_MEAN, full_matrices=False)
_LAB_COMPONENTS = _LAB_COMPONENTS[:2]
_LAB_PROJECTED = (_LAB_PROTOTYPES - _LAB_MEAN) @ _LAB_COMPONENTS.T


def classify_bgr_lab_pca(bgr: Iterable[int]) -> str | None:
    """Classifica a cor pelo vizinho mais próximo em Lab projetado por PCA.

    A distância é limitada para rejeitar pixels que não se parecem com uma
    faixa. Para cinzas/metal, a regra HSV existente continua sendo mais
    confiável e é usada como fallback.
    """

    values = list(bgr)
    pixel = np.uint8([[values]])
    hsv_name = classify_bgr(values)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    projected = (lab - _LAB_MEAN) @ _LAB_COMPONENTS.T
    distances = np.linalg.norm(_LAB_PROJECTED - projected, axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) > 55.0:
        return hsv_name
    name = _LAB_PROTOTYPE_NAMES[index]
    if name in {"black", "gray", "white", "silver"} and hsv_name is not None:
        return hsv_name
    return name


def decode_color_bands(bands: Iterable[str | None]) -> ResistanceValue | None:
    """Converte 4 ou 5 faixas no valor nominal e na tolerância."""

    normalized = tuple(color for color in bands if color in DIGITS or color in MULTIPLIERS)
    if len(normalized) not in (4, 5):
        return None

    try:
        if len(normalized) == 4:
            first, second, multiplier, tolerance = normalized
            significant = DIGITS[first] * 10 + DIGITS[second]
        else:
            first, second, third, multiplier, tolerance = normalized
            significant = DIGITS[first] * 100 + DIGITS[second] * 10 + DIGITS[third]
        if multiplier not in MULTIPLIERS or tolerance not in TOLERANCES:
            return None
    except (KeyError, ValueError):
        return None

    ohms = float(significant * (10 ** MULTIPLIERS[multiplier]))
    return ResistanceValue(
        ohms=int(ohms) if ohms.is_integer() else ohms,
        tolerance_percent=TOLERANCES[tolerance],
        bands=normalized,
    )
