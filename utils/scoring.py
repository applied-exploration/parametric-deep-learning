from functools import reduce
from .types import Instruction
import numpy as np


def score_instructions(
    y: list[list[Instruction]], pred: list[list[Instruction]]
) -> float:
    min_score = 0.0

    # check if instruction sequence is the same, if not, return min_score (dumb solution)
    instructions_match = reduce(
        lambda acc, x: acc and x, map(lambda x: type(x[0]) == type(x[1]), zip(y, pred))
    )
    if not instructions_match:
        return min_score

    def get_params(instructions: list[Instruction]) -> list[list[float]]:
        return [instruction.get_parameters() for instruction in instructions]

    y_params = [get_params(row) for row in y]
    pred_params = [get_params(row) for row in pred]
    return np.sum(
        np.abs([np.subtract(row[0], row[1]) for row in zip(y_params, pred_params)])
    ) / len(y)
