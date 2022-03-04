import torch
from data.types import Point, DataConfig
from typing import Callable


def embed_grid(dataconfig: DataConfig) -> Callable:
    def _embed_grid(input: list[int]) -> torch.Tensor:
        return torch.Tensor(input).view(dataconfig.canvas_size, dataconfig.canvas_size)

    return _embed_grid


def embed_and_normalize_points(dataconfig: DataConfig) -> Callable:
    def _embed_and_normalize_points(points: list[Point]):
        normalized = _normalize_to_canvas(points, dataconfig)
        return torch.flatten(torch.Tensor(normalized))

    return _embed_and_normalize_points


def _normalize_to_canvas(points: list[Point], data_config: DataConfig) -> list[Point]:
    return [
        (point[0] / data_config.canvas_size, point[1] / data_config.canvas_size)
        for point in points
    ]
