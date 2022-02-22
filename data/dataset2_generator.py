import pandas as pd
import random

from tqdm import tqdm
import numpy as np
from data.types import Circle, Modifiers, Primitives, Translate, Constraint, DataConfig
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


def generate_dataset2(config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]
    data = []
    all_features = []
    all_programs = []

    for _ in tqdm(range(config.dataset_size)):
        random_samples = []
        plain_samples = []

        primitives: Primitives = [
            Circle(random_radius(config), *random_translation(config))
            for _ in range(config.num_circles)
        ]

        modifiers: Modifiers = [
            Translate(*random_translation(config), index=0),
            Translate(*random_translation(config), index=1),
            Constraint(x=25, y=0, indicies=(0, 1)),
        ]

        """ 1. Apply modifiers """
        primitives, modifiers = render(primitives + modifiers)
        if display_plot:
            all_programs.append(primitives)

        """ 2. Sample operations """
        for _ in range(config.num_sample_points):
            for primitive in primitives:
                x, y = primitive.get_random_point()
                random_samples.extend([str(x) + " " + str(y) + "\n"])

                if display_plot:
                    plain_samples.extend([(x, y)])

        all_features.append(plain_samples)
        features_collapsed = "".join(random_samples)
        labels_collapsed = write_definition(primitives, modifiers)

        data.append([features_collapsed, labels_collapsed])

    if display_plot:
        num_to_display = min(15, len(all_features))
        display_both(
            all_features[:num_to_display], all_programs[:num_to_display], config
        )

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/dataset.csv", index=True)
