# import torch
from utils.parse import parse_program, parse_points
import numpy as np
import math
from typing import Tuple, List


def calculate_label_loss(estimated_program_values: list, actual_values: list) -> float:
    return np.sum(np.square(np.subtract(estimated_program_values, actual_values)))


def calculate_symbolic_loss(program: str, input_points: str) -> float:
    est_program: list = parse_program(program)
    parsed_input_points: list[tuple] = parse_points(input_points)
    x, y, r = run_program(est_program)

    loss = 0
    for point in parsed_input_points:
        distance = math.hypot(point[0] - x, point[1] - y)
        loss += (distance - r) ** 2
    return loss


def run_program(program: list) -> Tuple[float, float, float]:
    r = program[0].r
    x, y = program[1].x, program[1].y

    return x, y, r


if __name__ == "__main__":
    run_test()
