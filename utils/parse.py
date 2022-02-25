from data.types import (
    Instruction,
    Circle,
    Point,
    Translate,
    Constraint,
    Program,
    Triangle,
    Square,
    Rotate,
)


def parse_program(input: str) -> Program:
    def parse_line(line: str) -> Instruction:
        parameters = line.split(" ")
        if line.startswith("CIRCLE"):
            return Circle(
                r=float(parameters[1]),
                x=float(parameters[2]),
                y=float(parameters[3]),
                angle=float(parameters[4]),
            )
        elif line.startswith("SQUARE"):
            return Square(
                size=float(parameters[1]),
                x=float(parameters[2]),
                y=float(parameters[3]),
                angle=float(parameters[4]),
            )
        elif line.startswith("TRIANGLE"):
            return Triangle(
                size=float(parameters[1]),
                x=float(parameters[2]),
                y=float(parameters[3]),
                angle=float(parameters[4]),
            )
        elif line.startswith("TRANSLATION"):
            return Translate(
                x=float(parameters[1]), y=float(parameters[2]), index=int(parameters[3])
            )
        elif line.startswith("CONSTRAINT"):
            return Constraint(
                x=float(parameters[1]),
                y=float(parameters[2]),
                indicies=(int(parameters[3]), int(parameters[4])),
            )
        elif line.startswith("ROTATION"):
            return Rotate(
                angle=float(parameters[1]),
                index=int(parameters[2]),
            )
        else:
            raise Exception(f"Unknown instruction: {line}")

    return [parse_line(line) for line in input.splitlines()]


def parse_points(input: str) -> list[Point]:
    return [tuple(map(float, line.split(" "))) for line in input.splitlines()]
