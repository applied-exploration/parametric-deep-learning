#%%
from dataclasses import dataclass
from typing import Union


@dataclass
class Circle:
    r: float


@dataclass
class Translation:
    x: float
    y: float


Instruction = Union[Circle, Translation]
Point = tuple[float, float]


def parse_program(input: str) -> list[Instruction]:
    def parse_line(line: str) -> Instruction:
        if line.startswith("CIRCLE"):
            return Circle(float(line.split(" ")[1]))
        elif line.startswith("TRANSLATION"):
            return Translation(float(line.split(" ")[1]), float(line.split(" ")[2]))
        else:
            raise Exception(f"Unknown instruction: {line}")

    return [parse_line(line) for line in input.splitlines()]


instruction_set = {"CIRCLE": Circle, "TRANSLATION": Translation}


def parse_points(input: str) -> list[Point]:
    return [tuple(map(float, line.split(" "))) for line in input.splitlines()]


program = parse_program("CIRCLE 2.1\nTRANSLATION 1.2 3.4")
print(program)

points = parse_points("1 2\n3 4\n5 6\n7 8")
print(points)

# %%
