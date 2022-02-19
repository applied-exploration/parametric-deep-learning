import pandas as pd
import math
import random

from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from data.components import Circle, Translate, Constrain
from data.data_utils import DataConfig, display_circles

def random_radius(config: DataConfig) -> float:
    return min(config.max_radius, max(config.min_radius, random.random() * config.max_radius))

def random_translation(config:DataConfig) -> tuple[float, float]:
    return (
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
        random.uniform(-1, 1) * (config.canvas_size / 2 - config.max_radius),
    )

def generate_dataset2(config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]

    data = []
    all_circles = []
    for _ in tqdm(range(config.dataset_size)):
        random_samples = []
        plain_samples = []

        primitives = [Circle(random_radius(config), random_translation(config)) for _ in range(config.num_circles)]

        translate_1 = Translate(*random_translation(config))
        translate_2 = Translate(*random_translation(config))
        constrain_1 = Constrain(3, 0)
        
        """ 1. Apply instructions """
        primitives = translate_1.apply(primitives, 0)
        primitives = translate_2.apply(primitives, 1)
        primitives = constrain_1.apply(primitives, (0, 1))
         
        
        """ 2. Sample operations """
        for _ in range(config.num_sample_points):
            for primitive in primitives:
                x, y = primitive.get_random_point()
                random_samples.extend([str(x) + " " + str(y) + "\n"])

                if display_plot:
                    plain_samples.extend([(x, y)])

        all_circles.append(plain_samples)
        features_collapsed = "".join(random_samples)
        labels_collapsed = "".join(
            [
                "CIRCLE {}\n".format(random_circle_radius),
                "TRANSLATION {} {}".format(random_translation_x, random_translation_y),
            ]
        )

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
    )
    generate_dataset2(dataconfig, display_plot=False)
