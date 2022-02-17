import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class DataConfig:
    canvas_size: int
    max_radius: int
    num_sample_points: int
    num_circles: int


def random_circle_point(radius: float) -> tuple[float, float]:
    angle = random.random() * math.pi * 2
    x = math.cos(angle) * radius
    y = math.sin(angle) * radius
    return x, y


def translation(
    x: float, y: float, random_translation_x: float, random_translation_y: float
) -> tuple[float, float]:
    translated_x = x + random_translation_x
    translated_y = y + random_translation_y
    return translated_x, translated_y


def generate_dataset(config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]

    data = []
    for _ in tqdm(range(config.num_circles)):
        random_samples = []
        plain_samples = []

        random_circle_radius = min(
            config.max_radius, random.random() * config.max_radius
        )
        random_translation_x = random.uniform(-1, 1) * (
            config.canvas_size / 2 - config.max_radius
        )
        random_translation_y = random.uniform(-1, 1) * (
            config.canvas_size / 2 - config.max_radius
        )

        for j in range(config.num_sample_points):
            x, y = random_circle_point(random_circle_radius)
            x, y = translation(x, y, random_translation_x, random_translation_y)
            random_samples.extend([str(x) + " " + str(y) + "\n"])

            if display_plot:
                plain_samples.extend([(x, y)])

        features_collapsed = "".join(random_samples)
        labels_collapsed = "".join(
            [
                "CIRCLE {}\n".format(random_circle_radius),
                "TRANSLATION {} {}".format(random_translation_x, random_translation_y),
            ]
        )

        data.append([features_collapsed, labels_collapsed])
        ## Plot for reality check
        if display_plot:
            zipped = list(zip(*plain_samples))
            plt.gca().set_aspect("equal")
            plt.scatter(zipped[0], zipped[1])
            plt.xlim(-config.canvas_size / 2, config.canvas_size)
            plt.ylim(-config.canvas_size / 2, config.canvas_size)
            plt.show()

    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/dataset.csv", index=True)


if __name__ == "__main__":
    dataconfig = DataConfig(
        canvas_size=100, max_radius=20, num_sample_points=100, num_circles=500
    )

    generate_dataset(dataconfig, display_plot=False)
