import pandas as pd
import random
from tqdm import tqdm

from data.types import Circle, Translate, DataConfig
from render.utils import display_features


def generate_dataset1(config: DataConfig, display_plot: bool = False):
    column_names = ["features", "label"]

    data = []
    all_circles = []
    for _ in tqdm(range(config.num_circles)):
        random_samples = []
        plain_samples = []

        random_circle_radius = min(
            config.max_radius,
            max(config.min_radius, random.random() * config.max_radius),
        )
        random_translation_x = random.uniform(-1, 1) * (
            config.canvas_size / 2 - config.max_radius
        )
        random_translation_y = random.uniform(-1, 1) * (
            config.canvas_size / 2 - config.max_radius
        )

        new_circle = Circle(
            random_circle_radius, random_translation_x, random_translation_y
        )
        new_translate = Translate(random_translation_x, random_translation_y)

        for _ in range(config.num_sample_points):
            x, y = new_circle.get_random_point()
            x, y = new_translate.apply(x, y, random_translation_x, random_translation_y)
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
    generate_dataset1(dataconfig, display_plot=False)
