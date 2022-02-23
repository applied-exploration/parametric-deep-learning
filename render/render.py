from dis import Instruction
from data.types import Circle, Primitive, Primitives, Modifier, Modifiers, Program


def render(instructions: Program) -> tuple[Primitives, Modifiers]:

    primitives: Primitives = []
    modifiers: Modifiers = []

    for instruction in instructions:
        if isinstance(instruction, Primitive):
            primitives.append(instruction)
        elif isinstance(instruction, Modifier):
            modifiers.append(instruction)

    for modifier in modifiers:
        primitives = modifier.apply(primitives)

    return primitives, modifiers
