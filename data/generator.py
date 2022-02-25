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

from data.utils import write_definition, map_primitives, map_modifiers


def generator(name: str, config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]
    data, all_features, all_programs = [], [], []

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
