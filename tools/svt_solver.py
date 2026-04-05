from typing import Any, Dict

from tools.base_tool import Tool


class SVTSolver(Tool):
    name = "svt_solver"
    description = "Solves distance-speed-time equations given 2 of 3 quantities."

    def __init__(self, unit_converter):
        self.unit_converter = unit_converter

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.solve(
            distance=payload.get("distance"),
            speed=payload.get("speed"),
            time=payload.get("time"),
        )

    def solve(self, distance=None, speed=None, time=None) -> dict:
        known_count = sum(
            item is not None for item in [distance, speed, time]
        )

        if known_count != 2:
            return {
                "ok": False,
                "tool": self.name,
                "error": "Exactly two of distance, speed, and time must be known.",
            }

        normalized_distance = self.unit_converter.normalize_quantity(
            "distance", distance
        )
        normalized_speed = self.unit_converter.normalize_quantity(
            "speed", speed
        )
        normalized_time = self.unit_converter.normalize_quantity(
            "time", time
        )

        if normalized_speed is not None and normalized_speed["value"] == 0:
            return {
                "ok": False,
                "tool": self.name,
                "error": "Speed cannot be zero.",
            }

        if (
            normalized_time is not None
            and normalized_time["value"] == 0
            and distance is None
        ):
            return {
                "ok": False,
                "tool": self.name,
                "error": "Time cannot be zero when solving for distance or speed.",
            }

        if normalized_distance is None:
            solved_for = "distance"
            value_si = normalized_speed["value"] * normalized_time["value"]
            formula = "distance = speed * time"

        elif normalized_speed is None:
            if normalized_time["value"] == 0:
                return {
                    "ok": False,
                    "tool": self.name,
                    "error": "Time cannot be zero when solving for speed.",
                }
            solved_for = "speed"
            value_si = normalized_distance["value"] / normalized_time["value"]
            formula = "speed = distance / time"

        else:
            solved_for = "time"
            value_si = normalized_distance["value"] / normalized_speed["value"]
            formula = "time = distance / speed"

        formatted = self.unit_converter.format_human_readable(
            quantity_name=solved_for,
            value_si=value_si,
        )

        return {
            "ok": True,
            "tool": self.name,
            "solved_for": solved_for,
            "value_si": value_si,
            "unit_si": {
                "distance": "meter",
                "speed": "meter/second",
                "time": "second",
            }[solved_for],
            "display_value": formatted["value"],
            "display_unit": formatted["unit"],
            "human_readable": formatted["text"],
            "formula": formula,
            "inputs": {
                "distance": normalized_distance,
                "speed": normalized_speed,
                "time": normalized_time,
            },
        }
