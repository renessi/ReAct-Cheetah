from pint import UnitRegistry


class UnitConverter:
    def __init__(self):
        self.ureg = UnitRegistry()

        self.unit_aliases = {
            "км": "kilometer",
            "km": "kilometer",
            "м": "meter",
            "m": "meter",
            "км/ч": "kilometer/hour",
            "km/h": "kilometer/hour",
            "м/с": "meter/second",
            "m/s": "meter/second",
            "ч": "hour",
            "час": "hour",
            "часа": "hour",
            "часов": "hour",
            "h": "hour",
            "hr": "hour",
            "hour": "hour",
            "hours": "hour",
            "мин": "minute",
            "минута": "minute",
            "минуты": "minute",
            "минут": "minute",
            "min": "minute",
            "minute": "minute",
            "minutes": "minute",
            "сек": "second",
            "секунда": "second",
            "секунды": "second",
            "секунд": "second",
            "s": "second",
            "sec": "second",
            "second": "second",
            "seconds": "second",
        }

        self.base_units = {
            "distance": "meter",
            "speed": "meter/second",
            "time": "second",
        }

    def normalize_unit(self, raw_unit: str) -> str:
        unit = raw_unit.strip().lower()
        return self.unit_aliases.get(unit, unit)

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        normalized_from = self.normalize_unit(from_unit)
        normalized_to = self.normalize_unit(to_unit)

        quantity = value * self.ureg(normalized_from)
        converted = quantity.to(normalized_to)
        return float(converted.magnitude)

    def normalize_quantity(self, quantity_name: str, quantity_data: dict) -> dict:
        if quantity_data is None:
            return None

        target_unit = self.base_units[quantity_name]

        converted_value = self.convert(
            value=quantity_data["value"],
            from_unit=quantity_data["unit"],
            to_unit=target_unit,
        )

        return {
            "value": converted_value,
            "unit": target_unit,
        }

    def format_human_readable(self, quantity_name: str, value_si: float) -> dict:
        if quantity_name == "time":
            hours = int(value_si // 3600)
            minutes = int((value_si % 3600) // 60)
            seconds = round(value_si % 60, 1)
            if hours > 0 and minutes > 0:
                text = f"{hours} h {minutes} min"
            elif hours > 0:
                text = f"{hours} h"
            elif minutes > 0 and seconds > 0:
                text = f"{minutes} min {seconds} s"
            elif minutes > 0:
                text = f"{minutes} min"
            else:
                text = f"{seconds} s"
            if hours > 0:
                unit = "h"
            elif minutes > 0:
                unit = "min"
            else:
                unit = "s"
            return {"value": value_si, "unit": unit, "text": text}
        elif quantity_name == "distance":
            if value_si >= 1000:
                value = round(value_si / 1000.0, 2)
                unit = "km"
            else:
                value = round(value_si, 2)
                unit = "m"
            return {"value": value, "unit": unit, "text": f"{value} {unit}"}
        elif quantity_name == "speed":
            if value_si >= 1:
                value = round(value_si * 3.6, 2)
                unit = "km/h"
            else:
                value = round(value_si, 2)
                unit = "m/s"
            return {"value": value, "unit": unit, "text": f"{value} {unit}"}
        else:
            value = round(value_si, 2)