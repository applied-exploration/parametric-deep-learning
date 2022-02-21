from utils.parse import Instruction
import torch
from data.types import Circle, Constraint, Translate, all_instructions
from generate_dataset import DataConfig
from .utils import quantize, from_onehot, to_onehot
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
    The first three dimensions are one-hot encoded versions of the type.
    The next three dimensions are quantized continuous parameters (padded if not present).
    """
    quantize_bins = 128
    parameter_padding = torch.Tensor([0.0])
    index_padding = torch.Tensor([0.0] * dataconfig.max_definition_len)
    instruction_type = to_onehot(
        all_instructions[type(instruction)], len(all_instructions)
    )
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
        return torch.cat(
            [
                instruction_type,
                x,
                y,
                radius,
                index_padding,
                index_padding,
            ],
            dim=0,
        )
    elif isinstance(instruction, Translate):
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        return torch.cat(
            [
                instruction_type,
                x,
                y,
                parameter_padding,
                to_onehot(instruction.index, dataconfig.max_definition_len),
                index_padding,
            ],
            dim=0,
        )

    elif isinstance(instruction, Constraint):
        x = quantize(
            torch.Tensor([instruction.x / dataconfig.canvas_size]), quantize_bins
        )
        y = quantize(
            torch.Tensor([instruction.y / dataconfig.canvas_size]), quantize_bins
        )
        index1, index2 = instruction.indicies
        return torch.cat(
            [
                instruction_type,
                x,
                y,
                parameter_padding,
                to_onehot(index1, dataconfig.max_definition_len),
                to_onehot(index2, dataconfig.max_definition_len),
            ],
            dim=0,
        )

    else:
        raise Exception(f"Unknown instruction: {instruction}")


def from_embeddings_to_instructions(
    embeddings: torch.Tensor, dataconfig: DataConfig
) -> list[list[Instruction]]:
    assert embeddings.shape[1] % dataconfig.instruction_embedding_size == 0
    no_of_instructions = int(
        embeddings.shape[1] / dataconfig.instruction_embedding_size
    )
    embeddings = embeddings.cpu().detach().numpy()

    def single_embedding_to_instruction(
        embedding: np.ndarray, no_of_instruction_types: int
    ) -> Instruction:
        instruction_type = from_onehot(embedding[no_of_instructions])
        parameters_start_from = no_of_instruction_types

        if instruction_type == 0:
            return Circle(
                r=embedding[parameters_start_from] * dataconfig.max_radius,
                x=embedding[parameters_start_from + 1] * dataconfig.canvas_size,
                y=embedding[parameters_start_from + 2] * dataconfig.canvas_size,
            )
        elif instruction_type == 1:
            return Translate(
                x=embedding[parameters_start_from] * dataconfig.canvas_size,
                y=embedding[parameters_start_from + 1] * dataconfig.canvas_size,
                index=int(embedding[parameters_start_from + 2]),
            )
        elif instruction_type == 2:
            return Constraint(
                embedding[parameters_start_from] * dataconfig.canvas_size,
                embedding[parameters_start_from + 1] * dataconfig.canvas_size,
                (
                    int(embedding[parameters_start_from + 2]),
                    int(embedding[parameters_start_from + 3]),
                ),
            )
        else:
            raise Exception(f"Unknown instruction: {embedding}")

    return [
        [
            single_embedding_to_instruction(e, len(all_instructions))
            for e in np.array_split(row, no_of_instructions)
        ]
        for row in embeddings
    ]
