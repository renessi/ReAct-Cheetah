from tools.unit_converter import UnitConverter
from tools.svt_solver import SVTSolver


def test_solve_time_from_distance_and_speed():
    converter = UnitConverter()
    solver = SVTSolver(converter)

    result = solver.solve(
        distance={"value": 60, "unit": "km"},
        speed={"value": 120, "unit": "km/h"},
        time=None,
    )

    assert result["ok"] is True
    assert result["solved_for"] == "time"
    assert result["unit_si"] == "second"
    assert round(result["value_si"], 2) == 1800.0
    assert result["display_unit"] == "min"