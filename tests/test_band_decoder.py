from resistor_detector.band_decoder import value_to_bands


def test_value_to_four_band_sequence() -> None:
    label = value_to_bands(100_000, 4, 5.0)
    assert label.colors == ("brown", "black", "yellow", "gold")


def test_value_to_five_band_sequence() -> None:
    label = value_to_bands(4_700, 5, 1.0)
    assert label.colors == ("yellow", "violet", "black", "brown", "brown")
