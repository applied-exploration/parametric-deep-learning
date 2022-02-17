import torch
from utils.types import Point
from generate_dataset import DataConfig
from utils.embedding import normalize_to_canvas


def embed_and_normalize_points(
    dataconfig: DataConfig, points: list[Point]
) -> torch.Tensor:
    normalized = normalize_to_canvas(points, dataconfig)
    return torch.flatten(torch.Tensor(normalized))
