import sys
from typing import Literal
from pydantic import BaseModel
from loop import run, calc

FACTORS = {("mi", "km"): 1.609344, ("km", "mi"): 1 / 1.609344, ("lb", "kg"): 0.45359237, ("kg", "lb"): 1 / 0.45359237}


class ConvertArgs(BaseModel):
    value: float
    from_unit: Literal["mi", "km", "lb", "kg"]
    to_unit: Literal["mi", "km", "lb", "kg"]


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    if from_unit == to_unit:
        return str(value)
    return f"{value * FACTORS[(from_unit, to_unit)]:.4f} {to_unit}"


VAGUE = [
    {"type": "function", "function": {
        "name": "calc", "description": "Does calculations.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "convert_units", "description": "Converts values.",
        "parameters": ConvertArgs.model_json_schema()}},
]

PRECISE = [
    {"type": "function", "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression. Use it for all math; never do math in your head.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "convert_units",
        "description": "Convert a quantity between units. Use this for any unit conversion instead of multiplying by a factor yourself.",
        "parameters": ConvertArgs.model_json_schema()}},
]

REGISTRY = {"calc": calc, "convert_units": convert_units}

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many kilometres is 26.2 miles?"
    print("== vague descriptions ==")
    print(run(question, tools=VAGUE, registry=REGISTRY))
    print()
    print("== precise descriptions ==")
    print(run(question, tools=PRECISE, registry=REGISTRY))
