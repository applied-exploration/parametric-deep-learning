from utils.parse import Instruction
import torch
from data.types import Circle, Constraint, Translate, Point
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
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        return torch.Tensor([1.0, 0.0, 0.0, radius, x, y, padding])
    elif isinstance(instruction, Translate):
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        return torch.Tensor([0.0, 1.0, 0.0, x, y, instruction.index, padding])
    elif isinstance(instruction, Constraint):
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        index1, index2 = instruction.indicies
        return torch.Tensor([0.0, 0.0, 1.0, x, y, index1, index2])
    else:
        raise Exception(f"Unknown instruction: {instruction}")


def from_embeddings_to_instructions(
    embeddings: torch.Tensor, dataconfig: DataConfig
) -> list[list[Instruction]]:
    assert embeddings.shape[1] % dataconfig.instruction_embeddding_size == 0
    no_of_instructions = int(embeddings.shape[1] / dataconfig.instruction_embeddding_size)
    embeddings = embeddings.cpu().detach().numpy()

    def single_embedding_to_instruction(embedding: np.ndarray) -> Instruction:

        def softmax(x: np.ndarray) -> np.ndarray: return np.exp(x) / np.sum(np.exp(x))
        instruction_type = np.argmax(softmax(embedding[:3]))

        if instruction_type == 0:
            return Circle(r=embedding[3] * dataconfig.max_radius, x=embedding[4] * dataconfig.canvas_size, y=embedding[5] * dataconfig.canvas_size)
        elif instruction_type == 1:
            return Translate(
                x=embedding[3] * dataconfig.canvas_size,
                y=embedding[4] * dataconfig.canvas_size,
                index=int(embedding[5]),
            )
        elif instruction_type == 2:
            return Constraint(
                embedding[3] * dataconfig.canvas_size,
                embedding[4] * dataconfig.canvas_size,
                (int(embedding[5]), int(embedding[6])),
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
