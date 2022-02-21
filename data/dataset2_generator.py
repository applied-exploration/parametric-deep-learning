import pandas as pd
import random

from tqdm import tqdm
import numpy as np
from .types import Circle, Translate, Constraint
from .utils import DataConfig, display_circles
from typing import Union


def random_radius(config: DataConfig) -> float:
    return min(
        config.max_radius, max(config.min_radius, random.random() * config.max_radius)
    )


def random_translation(config: DataConfig) -> tuple[float, float]:
    return (
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
    )


def render(
    primitives: list[Circle], instructions: list[Union[Translate, Constraint]]
) -> tuple[list[Circle], list[Union[Translate, Constraint]]]:
    for instruction in instructions:
        primitives = instruction.apply(primitives)

    return primitives, instructions


def write_definition(
    primitives: list[Circle], instructions: list[Union[Translate, Constraint]]
) -> str:
    primitive_str = "".join(
        [
            "{} {} {} {}\n".format(primitive.name, *primitive.get_params())
            for primitive in primitives
        ]
    )
    instructions_str = "".join(
        [
            "{} {} {} {} {}\n".format(instruction.name, *instruction.get_params())
            for instruction in instructions
        ]
    )

    labels_collapsed = "".join([primitive_str, instructions_str])

    return labels_collapsed


def generate_dataset2(config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]
    data = []
    all_circles = []

    for _ in tqdm(range(config.dataset_size)):
        random_samples = []
        plain_samples = []

        primitives = [
            Circle(random_radius(config), *random_translation(config))
            for _ in range(config.num_circles)
        ]
        instructions = [
            Translate(*random_translation(config), index=0),
            Translate(*random_translation(config), index=1),
            Constraint(x=25, y=0, indicies=(0, 1)),
        ]

        """ 1. Apply instructions """
        primitives, instructions = render(primitives, instructions)

        """ 2. Sample operations """
        for _ in range(config.num_sample_points):
            for primitive in primitives:
                x, y = primitive.get_random_point()
                random_samples.extend([str(x) + " " + str(y) + "\n"])

                if display_plot:
                    plain_samples.extend([(x, y)])

        all_circles.append(plain_samples)
        features_collapsed = "".join(random_samples)
        labels_collapsed = write_definition(primitives, instructions)

        data.append([features_collapsed, labels_collapsed])

    if display_plot:
        display_circles(
            all_circles[: min(15, len(all_circles))], config
        )  ## Plot for reality check

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/dataset.csv", index=True)


if __name__ == "__main__":
    dataconfig = DataConfig(
        canvas_size=100,
        dataset_size=5,
        min_radius=10,
        max_radius=20,
        num_sample_points=100,
        num_circles=1,
        instruction_embedding_size=7,
    )
    generate_dataset2(dataconfig, display_plot=False)
