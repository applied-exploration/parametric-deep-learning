import pandas as pd
import random

from tqdm import tqdm
import numpy as np
from data.types import (
    Circle,
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
            "{} {} {} {}\n".format(primitive.name, *primitive.get_params())
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


def generator(name: str, config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]
    data = []
    all_features = []
    all_programs = []

    for _ in tqdm(range(config.dataset_size)):
        plain_samples = []

        primitives_to_use = random.choices(
            config.primitive_types, k=config.num_primitives
        )
        modifiers_to_use = random.choices(config.modifier_types, k=config.num_modifiers)

        primitives: Primitives = []
        modifiers: Modifiers = []
        for primitive in primitives_to_use:
            if type(primitive) == Circle:
                new_primitive = Circle(
                    random_radius(config), *random_translation(config)
                )
            else:
                new_primitive = Circle(
                    random_radius(config), *random_translation(config)
                )
            primitives.append(new_primitive)

        for modifier in modifiers_to_use:
            if type(modifier) == Translate:
                new_modifier = Translate(*random_translation(config), index=0)
            elif type(modifier) == Rotate:
                new_modifier = Rotate(-180, index=1)
            else:  # type(modifier) == Constraint:
                new_modifier = Constraint(x=25, y=0, indicies=(0, 1))

            modifiers.append(new_modifier)

        """ 1. Apply modifiers """
        instructions: list[Instruction] = [*primitives, *modifiers]
        primitives, modifiers = render(instructions)
        if display_plot:
            all_programs.append(primitives)

        """ 2. Sample operations """
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
        display_both(all_features[0], all_programs[0], config)

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/{}.csv".format(name), index=True)
