from dis import Instruction
from data.types import Circle, Primitives, Modifiers, Program


def render(instructions: Program) -> tuple[Primitives, Modifiers]:

    primitives: Primitives = []
    modifiers: Modifiers = []

    for instruction in instructions:
        if isinstance(instruction, Circle):
            primitives.append(instruction)
        else:
            modifiers.append(instruction)

    for modifier in modifiers:
        primitives = modifier.apply(primitives)

    return primitives, modifiers
