from tools.unit_converter import UnitConverter


def test_convert_kmh_to_mps():
    converter = UnitConverter()
    result = converter.convert(120, "km/h", "meter/second")

    assert round(result, 2) == 33.33


def test_convert_minutes_to_seconds():
    converter = UnitConverter()
    result = converter.convert(30, "min", "second")

    assert result == 1800.0