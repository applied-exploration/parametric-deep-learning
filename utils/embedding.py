import numpy as np
from utils.types import Point
from generate_dataset import DataConfig


def normalize_to_unit_vector(points: list[float]) -> list[float]:
    x = np.array(points)
    x = x / np.sqrt(x.dot(x))

    return x


def normalize_to_range(points: list[float]) -> np.ndarray:
    x = np.array(points)
    x = np.subtract(x, np.min(x))
    x = np.divide(x, np.max(x) - np.min(x))

    return x


def normalize_to_canvas(points: list[Point], data_config: DataConfig) -> list[Point]:
    return [
        (point[0] / data_config.canvas_size, point[1] / data_config.canvas_size)
        for point in points
    ]
