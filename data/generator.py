import pandas as pd
import random

from tqdm import tqdm
import numpy as np
from data.types import (
    Modifiers,
    Primitives,
    Translate,
    Constraint,
    DataConfig,
    Instruction,
)
from render.utils import display_both
from render.render import render

from data.utils import write_definition, map_primitives, map_modifiers


def random_dataset(config: DataConfig):
    primitives_to_use = random.choices(config.primitive_types, k=config.num_primitives)
    modifiers_to_use = random.choices(config.modifier_types, k=config.num_modifiers)

    primitives: Primitives = map_primitives(config, primitives_to_use)
    modifiers: Modifiers = map_modifiers(config, modifiers_to_use)

    return primitives, modifiers


def faces_dataset(config: DataConfig):
    primitives_to_use = random.choices(config.primitive_types, k=config.num_primitives)
    modifiers_to_use = config.modifier_types

    primitives: Primitives = map_primitives(
        config,
        primitives_to_use,
        initial_translation=[(0, 0) for _ in range(config.num_primitives)],
        initial_size=[config.max_radius for _ in range(config.num_primitives)],
    )
    distance_eyes = max(
        config.max_radius * 1.1, random.uniform(0.25, 0.75) * config.canvas_size / 2
    )
    distance_mouth = random.uniform(-1.2, -1.5) * distance_eyes
    modifiers: Modifiers = [
        Translate(
            random.uniform(-1, 1) * ((config.canvas_size / 2) - distance_eyes),
            random.uniform(-1, 0) * (distance_mouth - config.max_radius),
            0,
        ),
        Constraint(
            distance_eyes,
            0.0,
            (0, 1),
        ),
        Constraint(
            -distance_eyes / 2,
            distance_mouth,
            (1, 2),
        ),
    ]  # map_modifiers(config, modifiers_to_use)

    return primitives, modifiers


def choose_structure(config: DataConfig):
    if config.name == "faces":
        return faces_dataset(config)
    else:
        return random_dataset(config)


def generator(config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]
    data, all_features, all_programs = [], [], []

    if config.random_primitives:
        config.num_primitives = random.choice(range(2, config.num_primitives))

    for _ in tqdm(range(config.dataset_size)):
        """0. Prepare the primitives and modifiers"""
        primitives, modifiers = choose_structure(config)

        """ 1. Apply modifiers """
        instructions: list[Instruction] = [*primitives, *modifiers]
        primitives, modifiers = render(instructions)

        """ 2. Sample operations """
        img = None
        for primitive in primitives:
            if img is None:
                img = primitive.get_grid(config.canvas_size, config.canvas_size)
            else:
                img = np.maximum(
                    img, primitive.get_grid(config.canvas_size, config.canvas_size)
                )

        all_features.append(img)
        features_collapsed: str = " ".join(
            [" ".join([str(item) for item in row]) for row in img]
        )
        labels_collapsed: str = write_definition(primitives, modifiers)

        if display_plot:
            all_programs.append((primitives, labels_collapsed))
        data.append([features_collapsed, labels_collapsed])

    if display_plot:
        display_both(all_features, all_programs, config, interactive=True)

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/generated/{}.csv".format(config.name), index=True)
