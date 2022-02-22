from data.types import Circle, Translate, Constraint, DataConfig
from typing import Union


def render(
    primitives: list[Circle], instructions: list[Union[Translate, Constraint]]
) -> tuple[list[Circle], list[Union[Translate, Constraint]]]:
    for instruction in instructions:
        primitives = instruction.apply(primitives)

    return primitives, instructions
