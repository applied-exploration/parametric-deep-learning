from utils.parse import Instruction
import torch
from utils.types import Circle, Translation, Point
from generate_dataset import DataConfig
from utils.quantize import quantize


def embed_instructions(
    dataconfig: DataConfig, instructions: list[Instruction]
) -> torch.Tensor:
    """Embed an instruction into a tensor.
    The first two dimensions are one-hot encoded versions of the type.
    The last dimensions are embedded parameters (padded if not present).
    """
    return torch.stack(
        [__embed_instruction(dataconfig, instruction) for instruction in instructions], dim = 0
    )


def __embed_instruction(
    dataconfig: DataConfig, instruction: Instruction
) -> torch.Tensor:
    """Embed an instruction into a tensor.
    The first two dimensions are one-hot encoded versions of the type.
    The last dimensions are embedded parameters (padded if not present).
    """
    quantize_bins = 128
    padding = 0.0
    if isinstance(instruction, Circle):
        radius = quantize(
            torch.Tensor([instruction.r / dataconfig.max_radius]), quantize_bins
        )
        return torch.Tensor([0.0, 1.0, instruction.r, padding])
    elif isinstance(instruction, Translation):
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        return torch.Tensor([1.0, 0.0, x, y])
    else:
        raise Exception(f"Unknown instruction: {instruction}")
