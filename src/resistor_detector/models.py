from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResistanceValue:
    """Valor calculado a partir das faixas de um resistor."""

    ohms: float
    tolerance_percent: float | None
    bands: tuple[str, ...]

    @property
    def formatted(self) -> str:
        value = self.ohms
        if value >= 1_000_000:
            return f"{value / 1_000_000:g} MΩ"
        if value >= 1_000:
            return f"{value / 1_000:g} kΩ"
        return f"{value:g} Ω"


@dataclass(frozen=True)
class ResistorDetection:
    """Resultado de um resistor encontrado em um quadro."""

    bbox: tuple[int, int, int, int]
    bands: tuple[str, ...]
    value: ResistanceValue | None
    confidence: float

    @property
    def label(self) -> str:
        if self.value is None:
            return "resistor (valor não identificado)"
        tolerance = ""
        if self.value.tolerance_percent is not None:
            tolerance = f" ±{self.value.tolerance_percent:g}%"
        return f"{self.value.formatted}{tolerance}"

