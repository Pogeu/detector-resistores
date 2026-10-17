"""Ferramentas para detectar e identificar resistores em imagens e vídeos."""

from .colors import decode_color_bands
from .detection import analyze_frame

__all__ = ["analyze_frame", "decode_color_bands"]

