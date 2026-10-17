import numpy as np

from resistor_detector.automatic_component_analyzer import (
    analyze_frame,
    decode_4_band,
)


def test_decode_4_band_matches_upstream_maps():
    bands = [{"color": color} for color in ("red", "red", "brown", "gold")]
    assert decode_4_band(bands) == (220, "5%")


def test_headless_analyzer_returns_auditable_result():
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    result = analyze_frame(image)
    assert set(("bands", "colors", "decoded_results", "rejected")) <= result.keys()
