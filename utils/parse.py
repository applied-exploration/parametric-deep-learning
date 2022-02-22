from data.types import Instruction, Circle, Point, Translate, Constraint


def parse_program(input: str) -> list[Instruction]:
    def parse_line(line: str) -> Instruction:
        parameters = line.split(" ")
        if line.startswith("CIRCLE"):
            return Circle(
                float(parameters[1]), float(parameters[2]), float(parameters[3])
            )
        elif line.startswith("TRANSLATION"):
            return Translate(
                float(parameters[1]), float(parameters[2]), int(parameters[3])
            )
        elif line.startswith("CONSTRAINT"):
            return Constraint(
                float(parameters[1]),
                float(parameters[2]),
                (int(parameters[3]), int(parameters[4])),
            )
        else:
            raise Exception(f"Unknown instruction: {line}")

    return [parse_line(line) for line in input.splitlines()]


def parse_points(input: str) -> list[Point]:
    return [tuple(map(float, line.split(" "))) for line in input.splitlines()]
