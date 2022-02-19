from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class DataConfig:
    canvas_size: int
    min_radius: int
    max_radius: int
    num_sample_points: int
    num_circles: int


def display_circles(circles: list, config: DataConfig) -> None:

    plt.xlim(-config.canvas_size / 2, config.canvas_size)
    plt.ylim(-config.canvas_size / 2, config.canvas_size)
    plt.gca().set_aspect("equal")

    for circle in circles:
        zipped = list(zip(*circle))
        plt.scatter(zipped[0], zipped[1])
    plt.show()
