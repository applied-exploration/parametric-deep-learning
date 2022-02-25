import random

from data.types import (
    Circle,
    Square,
    Triangle,
    Modifiers,
    Primitives,
    Translate,
    Rotate,
    Constraint,
    DataConfig,
    Program,
    Instruction,
)



def _random_radius(config: DataConfig) -> float:
    return min(
        config.max_radius, max(config.min_radius, random.random() * config.max_radius)
    )


def _random_translation(config: DataConfig) -> tuple[float, float]:
    return (
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
    )

def _random_index(num_primitives: int) -> int:
    return random.randint(0, num_primitives - 1)


def _random_indecies(num_primitives: int) -> tuple[int, int]:
    sample = random.sample(range(num_primitives), k=2)
    return (sample[0], sample[1])


def write_definition(primitives: Primitives, modifiers: Modifiers) -> str:
    primitive_str = "".join(
        [
            "{} {} {} {} {}\n".format(primitive.name, *primitive.get_params())
            for primitive in primitives
        ]
    )
    modifiers_str = "".join(
        [
            "{} {} {} {} {}\n".format(instruction.name, *instruction.get_params())
            for instruction in modifiers
        ]
    )

    labels_collapsed = "".join([primitive_str, modifiers_str])

    return labels_collapsed


def map_primitives(config: DataConfig, primitives_to_use: list) -> Primitives:
    primitives: Primitives = []
    for primitive in primitives_to_use:
        if isinstance(Circle(0, 0, 0, 0), primitive):
            new_primitive = Circle(
                _random_radius(config), *_random_translation(config), 0
            )
        elif isinstance(Square(0, 0, 0, 0), primitive):
            new_primitive = Square(
                _random_radius(config), *_random_translation(config), 0
            )
        elif isinstance(Triangle(0, 0, 0, 0), primitive):
            new_primitive = Triangle(
                _random_radius(config), *_random_translation(config), 0
            )
        else:
            continue
        primitives.append(new_primitive)
    return primitives


def map_modifiers(config: DataConfig, modifiers_to_use: list) -> Modifiers:
    modifiers: Modifiers = []

    constraint_present = None
    for i, modifier in enumerate(modifiers_to_use):
        if isinstance(Constraint(0, 0, (0, 0)), modifier):
            constraint_present = Constraint(
                x=random.random() * config.canvas_size / 3,
                y=0,
                indicies=_random_indecies(config.num_primitives),
            )
            continue

        elif isinstance(Rotate(0, 0), modifier):
            new_modifier = Rotate(
                random.random() * 360, index=_random_index(config.num_primitives)
            )
        elif isinstance(Translate(0, 0, 0), modifier):
            new_modifier = Translate(
                *_random_translation(config), index=_random_index(config.num_primitives)
            )
        else:
            continue
        modifiers.append(new_modifier)

    if constraint_present is not None:
        modifiers.append(constraint_present)  # place modifiers as the last object

    return modifiers