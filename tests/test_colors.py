from resistor_detector.colors import decode_color_bands


def test_decode_four_band_resistor():
    value = decode_color_bands(("yellow", "violet", "orange", "gold"))

    assert value is not None
    assert value.ohms == 47_000
    assert value.tolerance_percent == 5
    assert value.formatted == "47 kΩ"


def test_decode_five_band_resistor():
    value = decode_color_bands(("brown", "black", "black", "red", "brown"))

    assert value is not None
    assert value.ohms == 10_000
    assert value.tolerance_percent == 1


def test_invalid_band_count_returns_none():
    assert decode_color_bands(("red", "red", "red")) is None

