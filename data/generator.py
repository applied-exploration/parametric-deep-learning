import pandas as pd
import random

from tqdm import tqdm
import numpy as np
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
from render.utils import display_features, display_program, display_both
from render.render import render
from typing import Union
from itertools import chain


def random_radius(config: DataConfig) -> float:
    return min(
        config.max_radius, max(config.min_radius, random.random() * config.max_radius)
    )


def random_translation(config: DataConfig) -> tuple[float, float]:
    return (
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
    )


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


def random_index(num_primitives: int) -> int:
    return random.randint(0, num_primitives - 1)


def random_indecies(num_primitives: int) -> tuple[int, int]:
    sample = random.sample(range(num_primitives), k=2)
    return (sample[0], sample[1])


def map_primitives(config: DataConfig, primitives_to_use: list) -> Primitives:
    primitives: Primitives = []
    for primitive in primitives_to_use:
        if isinstance(Circle(0, 0, 0, 0), primitive):
            new_primitive = Circle(
                random_radius(config), *random_translation(config), 0
            )
        elif isinstance(Square(0, 0, 0, 0), primitive):
            new_primitive = Square(
                random_radius(config), *random_translation(config), 0
            )
        elif isinstance(Triangle(0, 0, 0, 0), primitive):
            new_primitive = Triangle(
                random_radius(config), *random_translation(config), 0
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
                indicies=random_indecies(config.num_primitives),
            )
            continue

        elif isinstance(Rotate(0, 0), modifier):
            new_modifier = Rotate(
                random.random() * 360, index=random_index(config.num_primitives)
            )
        elif isinstance(Translate(0, 0, 0), modifier):
            new_modifier = Translate(
                *random_translation(config), index=random_index(config.num_primitives)
            )
        else:
            continue
        modifiers.append(new_modifier)

    if constraint_present is not None:
        modifiers.append(constraint_present)  # place modifiers as the last object

    return modifiers


def generator(name: str, config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]
    data = []
    all_features = []
    all_programs = []
    if config.random_primitives:
        config.num_primitives = random.choice(range(2, config.num_primitives))

    for _ in tqdm(range(config.dataset_size)):
        """0. Prepare the primitives and modifiers"""
        primitives_to_use = random.choices(
            config.primitive_types, k=config.num_primitives
        )
        modifiers_to_use = random.choices(config.modifier_types, k=config.num_modifiers)

        primitives: Primitives = map_primitives(config, primitives_to_use)
        modifiers: Modifiers = map_modifiers(config, modifiers_to_use)

        """ 1. Apply modifiers """
        instructions: list[Instruction] = [*primitives, *modifiers]
        primitives, modifiers = render(instructions)
        if display_plot:
            all_programs.append(primitives)

        """ 2. Sample operations """
        plain_samples = []
        for _ in range(config.num_sample_points):
            for primitive in primitives:
                x, y = primitive.get_random_point()
                plain_samples.extend([(x, y)])

        all_features.append(plain_samples)
        features_collapsed = "".join(
            [str(x) + " " + str(y) + "\n" for x, y in plain_samples]
        )
        labels_collapsed = write_definition(primitives, modifiers)

        data.append([features_collapsed, labels_collapsed])

    if display_plot:
        display_both(all_features, all_programs, config, interactive=True)

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/{}.csv".format(name), index=True)
