from utils.parse import Instruction
import torch
from utils.types import Circle, Translation, Point
from generate_dataset import DataConfig
from utils.quantize import quantize
import numpy as np


def embed_instructions(
    dataconfig: DataConfig, instructions: list[Instruction]
) -> torch.Tensor:
    """Embed an instruction into a tensor.
    The first two dimensions are one-hot encoded versions of the type.
    The last dimensions are embedded parameters (padded if not present).
    """
    return torch.flatten(
        torch.stack(
            [
                __embed_instruction(dataconfig, instruction)
                for instruction in instructions
            ],
            dim=0,
        )
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
        return torch.Tensor([0.0, radius, padding])
    elif isinstance(instruction, Translation):
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        return torch.Tensor([1.0, x, y])
    else:
        raise Exception(f"Unknown instruction: {instruction}")


def from_embeddings_to_instructions(
    embeddings: torch.Tensor, dataconfig: DataConfig
) -> list[list[Instruction]]:
    assert embeddings.shape[1] % 3 == 0
    no_of_instructions = int(embeddings.shape[1] / 3)
    embeddings = embeddings.cpu().detach().numpy()

    def single_embedding_to_instruction(embedding: np.ndarray) -> Instruction:
        if embedding[0] < 0.5:
            return Circle(embedding[1] * dataconfig.max_radius)
        elif embedding[0] > 0.5:
            return Translation(
                embedding[1] * dataconfig.canvas_size,
                embedding[2] * dataconfig.canvas_size,
            )
        else:
            raise Exception(f"Unknown instruction: {embedding}")

    return [
        [
            single_embedding_to_instruction(e)
            for e in np.array_split(row, no_of_instructions)
        ]
        for row in embeddings
    ]
