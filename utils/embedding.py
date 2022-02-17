from utils.parse import Instruction
import numpy as np
import torch
from utils.types import Circle, Translation, Point
from generate_dataset import DataConfig
from .quantize import quantize


def normalize_to_unit_vector(points: list[float]) -> list[float]:
    x = np.array(points)
    x = x / np.sqrt(x.dot(x))

    return x


def normalize_to_range(points: list[float]) -> np.ndarray:
    x = np.array(points)
    x = np.subtract(x, np.min(x))
    x = np.divide(x, np.max(x) - np.min(x))

    return x


def embed_instruction(dataconfig: DataConfig, instruction: Instruction) -> torch.Tensor:
    """Embed an instruction into a tensor.
    The first two dimensions are one-hot encoded versions of the type.
    The last dimensions are embedded parameters (padded if not present).
    """
    quantize_bins = 128
    padding = 0.0
    if isinstance(instruction, Circle):
        radius = quantize(instruction.r / dataconfig.max_radius, quantize_bins)
        return torch.Tensor([0, 1, instruction.r, padding])
    elif isinstance(instruction, Translation):
        x = quantize(instruction.x / dataconfig.canvas_size, quantize_bins)
        y = quantize(instruction.y / dataconfig.canvas_size, quantize_bins)
        return torch.Tensor([1, 0, x, y])
    else:
        raise Exception(f"Unknown instruction: {instruction}")
