from grpc import Call
from utils.parse import Instruction
import torch
from data.types import Circle, Constraint, Program, Translate, all_instructions
from generate_dataset import DataConfig
from .utils import quantize, from_onehot, to_onehot
import numpy as np
from typing import Callable


def embed_instructions(dataconfig: DataConfig) -> Callable:
    def _embed_instructions(instructions: Program) -> torch.Tensor:
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

    return _embed_instructions


def __embed_instruction(
    dataconfig: DataConfig, instruction: Instruction
) -> torch.Tensor:
    """Embed an instruction into a tensor.
    1. The first three dimensions are one-hot encoded versions of the type.
    2. The next three dimensions are quantized continuous parameters (padded if not present).
    3. The (2 * max_definition_len) dimensions are indicies of nodes (parameters to constraint) one-hot encoded.

    [0,0,1,  0.1,0.2,0.3,  0,0,0,0,0,0,0,0,0,1 0,0,0,0,0,0,0,0,1,0]
       1.         2.                        3.
    """
    quantize_bins = 100
    parameter_padding = torch.Tensor([0.0])
    index_padding = torch.Tensor([0.0] * dataconfig.max_definition_len)
    instruction_type = to_onehot(
        all_instructions[type(instruction)], len(all_instructions)
    )
    if isinstance(instruction, Circle):
        radius = quantize(instruction.r, dataconfig.max_radius)
        x = quantize(instruction.x, dataconfig.canvas_size)
        y = quantize(instruction.y, dataconfig.canvas_size)
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
        x = quantize(instruction.x, dataconfig.canvas_size)
        y = quantize(instruction.y, dataconfig.canvas_size)
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
        x = quantize(instruction.x, dataconfig.canvas_size)
        y = quantize(instruction.y, dataconfig.canvas_size)
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


def from_embeddings_to_instructions(dataconfig: DataConfig) -> Callable:
    def _from_embeddings_to_instructions(
        embeddings: torch.Tensor,
    ) -> list[list[Instruction]]:
        assert embeddings.shape[1] % dataconfig.instruction_embedding_size == 0
        no_of_instructions = int(
            embeddings.shape[1] / dataconfig.instruction_embedding_size
        )
        embeddings = embeddings.cpu().detach().numpy()

        def single_embedding_to_instruction(
            embedding: np.ndarray, no_of_instruction_types: int
        ) -> Instruction:
            instruction_type = from_onehot(embedding[:no_of_instruction_types])
            parameters_start_from = no_of_instruction_types

            if instruction_type == 0:
                return Circle(
                    x=embedding[parameters_start_from] * dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1] * dataconfig.canvas_size,
                    r=embedding[parameters_start_from + 2] * dataconfig.max_radius,
                )
            elif instruction_type == 1:
                return Translate(
                    x=embedding[parameters_start_from] * dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1] * dataconfig.canvas_size,
                    index=from_onehot(
                        embedding[
                            parameters_start_from
                            + 3 : parameters_start_from
                            + 3
                            + dataconfig.max_definition_len,
                        ]
                    ),
                )
            elif instruction_type == 2:
                indicies_start_from = parameters_start_from + 3
                return Constraint(
                    embedding[parameters_start_from] * dataconfig.canvas_size,
                    embedding[parameters_start_from + 1] * dataconfig.canvas_size,
                    (
                        from_onehot(
                            embedding[
                                indicies_start_from : indicies_start_from
                                + dataconfig.max_definition_len,
                            ]
                        ),
                        from_onehot(
                            embedding[
                                indicies_start_from
                                + dataconfig.max_definition_len : indicies_start_from
                                + (dataconfig.max_definition_len * 2),
                            ]
                        ),
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

    return _from_embeddings_to_instructions
