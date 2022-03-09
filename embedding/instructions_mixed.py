from utils.parse import Instruction
import torch
from data.types import (
    Circle,
    Constraint,
    Program,
    Translate,
    Triangle,
    Square,
    Rotate,
)
from generate_dataset import DataConfig
from .utils import quantize, from_onehot, to_onehot
import numpy as np
from .types import ProgramStaticEmbeddings
from torch import Tensor
from torch.nn import functional as F


class MixedProgramStaticEmbeddings(ProgramStaticEmbeddings):
    def __init__(self, dataconfig: DataConfig) -> None:
        self.dataconfig = dataconfig
        self.single_instruction_size = (
            len(dataconfig.instructions_map)
            + 3
            + ((dataconfig.num_primitives + 1) * 2)
            + 1
        )

    def program_to_tensor(self, program: Program) -> torch.Tensor:
        """Embed an instruction into a tensor.
        The first two dimensions are one-hot encoded versions of the type.
        The last dimensions are embedded parameters (padded if not present).
        """
        return torch.flatten(
            torch.stack(
                [
                    embed_instruction(self.dataconfig, instruction)
                    for instruction in program
                ],
                dim=0,
            )
        )

    def tensor_to_programs(self, tensor: Tensor) -> list[Program]:
        assert tensor.shape[1] % self.single_instruction_size == 0
        no_of_instructions = int(tensor.shape[1] / self.single_instruction_size)
        tensor = tensor.cpu().detach().numpy()

        def single_embedding_to_instruction(
            embedding: np.ndarray, no_of_instruction_types: int
        ) -> Instruction:
            instruction_type = from_onehot(embedding[:no_of_instruction_types])
            parameters_start_from = no_of_instruction_types

            if instruction_type == self.dataconfig.instructions_map[Circle]:
                return Circle(
                    x=embedding[parameters_start_from] * self.dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1]
                    * self.dataconfig.canvas_size,
                    r=embedding[parameters_start_from + 2] * self.dataconfig.max_radius,
                    angle=embedding[-1] * 360,
                )
            elif instruction_type == self.dataconfig.instructions_map[Square]:
                return Square(
                    x=embedding[parameters_start_from] * self.dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1]
                    * self.dataconfig.canvas_size,
                    size=embedding[parameters_start_from + 2]
                    * self.dataconfig.max_radius,
                    angle=embedding[-1] * 360,
                )
            elif instruction_type == self.dataconfig.instructions_map[Triangle]:
                return Triangle(
                    x=embedding[parameters_start_from] * self.dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1]
                    * self.dataconfig.canvas_size,
                    size=embedding[parameters_start_from + 2]
                    * self.dataconfig.max_radius,
                    angle=embedding[-1] * 360,
                )
            elif instruction_type == self.dataconfig.instructions_map[Translate]:
                return Translate(
                    x=embedding[parameters_start_from] * self.dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1]
                    * self.dataconfig.canvas_size,
                    index=from_onehot(
                        embedding[
                            parameters_start_from
                            + 3 : parameters_start_from
                            + 3
                            + self.dataconfig.num_primitives
                            + 1,
                        ]
                    ),
                )
            elif instruction_type == self.dataconfig.instructions_map[Constraint]:
                indicies_start_from = parameters_start_from + 3
                return Constraint(
                    x=embedding[parameters_start_from] * self.dataconfig.canvas_size,
                    y=embedding[parameters_start_from + 1]
                    * self.dataconfig.canvas_size,
                    indicies=(
                        from_onehot(
                            embedding[
                                indicies_start_from : indicies_start_from
                                + self.dataconfig.num_primitives
                                + 1,
                            ]
                        ),
                        from_onehot(
                            embedding[
                                indicies_start_from
                                + self.dataconfig.num_primitives
                                + 1 : indicies_start_from
                                + ((self.dataconfig.num_primitives + 1) * 2),
                            ]
                        ),
                    ),
                )
            elif instruction_type == self.dataconfig.instructions_map[Rotate]:
                indicies_start_from = parameters_start_from + 3
                return Rotate(
                    index=from_onehot(
                        embedding[
                            indicies_start_from : indicies_start_from
                            + self.dataconfig.num_primitives
                            + 1,
                        ]
                    ),
                    angle=embedding[-1] * 360,
                )
            else:
                raise Exception(f"Unknown instruction: {embedding}")

        return [
            [
                single_embedding_to_instruction(
                    e, len(self.dataconfig.instructions_map)
                )
                for e in np.array_split(row, no_of_instructions)
            ]
            for row in tensor
        ]

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape[1] % self.single_instruction_size == 0
        assert input.shape[0] == target.shape[0]

        no_of_instructions = int(input.shape[1] / self.single_instruction_size)
        input_per_instruction = input.view(
            input.shape[0], no_of_instructions, self.single_instruction_size
        )
        target_per_instruction = target.view(
            target.shape[0], no_of_instructions, self.single_instruction_size
        )

        index_instructions_end = len(self.dataconfig.instructions_map)
        input_instruction_types = input_per_instruction[:, :, :index_instructions_end]
        target_instruction_types = target_per_instruction[:, :, :index_instructions_end]
        loss_instructions_types = F.cross_entropy(
            input_instruction_types,
            target_instruction_types,
        )

        index_parameters_end = index_instructions_end + 3
        input_parameters = input_per_instruction[
            :, :, index_instructions_end:index_parameters_end
        ]
        target_parameters = target_per_instruction[
            :, :, index_instructions_end:index_parameters_end
        ]
        padding_mask = (target_parameters != -1.0).int()
        input_parameters = input_parameters * padding_mask
        target_parameters = target_parameters * padding_mask
        loss_parameters = F.mse_loss(input_parameters, target_parameters) * 50

        index_index1_end = index_parameters_end + self.dataconfig.num_primitives
        input_index1 = input_per_instruction[
            :, :, index_parameters_end:index_index1_end
        ]
        target_index1 = target_per_instruction[
            :, :, index_parameters_end:index_index1_end
        ]

        loss_index1 = F.cross_entropy(
            input_index1.view(-1, self.dataconfig.num_primitives),
            torch.argmax(target_index1.view(-1, self.dataconfig.num_primitives), dim=1),
            ignore_index=0,
        )

        index_index2_end = index_parameters_end + (self.dataconfig.num_primitives * 2)
        input_index2 = input_per_instruction[:, :, index_index1_end:index_index2_end]
        target_index2 = target_per_instruction[:, :, index_index1_end:index_index2_end]
        loss_index2 = F.cross_entropy(
            input_index2.view(-1, self.dataconfig.num_primitives),
            torch.argmax(target_index2.view(-1, self.dataconfig.num_primitives), dim=1),
            ignore_index=0,
        )

        index_angle_end = index_index2_end + 1
        input_angle = input_per_instruction[:, :, index_index2_end:index_angle_end]
        target_angle = target_per_instruction[:, :, index_index2_end:index_angle_end]
        padding_mask = (target_angle != -1.0).int()
        input_angle = input_angle * padding_mask
        target_angle = target_angle * padding_mask
        loss_angle = F.mse_loss(input_angle, target_angle) * 50

        return (
            loss_instructions_types
            + loss_parameters
            + loss_index1
            + loss_index2
            + loss_angle
        )


def embed_instruction(dataconfig: DataConfig, instruction: Instruction) -> torch.Tensor:
    """Embed one instruction into a tensor.
    1. The first n dimensions are one-hot encoded versions of the type.
    2. The next three dimensions are quantized continuous parameters (padded if not present).
    3. The (2 * num_primitives + 1) dimensions are indicies of nodes (parameters to constraint) one-hot encoded.
    4. The last dimension is quantized angle parameters (padded if not present).

    [0,0,1,  0.1,0.2,0.3,  0,0,0,0,0,0,0,0,0,1 0,0,0,0,0,0,0,0,1,0,  0.05]
       1.         2.                        3.                         4.
    """
    parameter_padding = torch.Tensor([-1.0])
    index_padding = to_onehot(0, dataconfig.num_primitives + 1)
    instruction_type = to_onehot(
        dataconfig.instructions_map[type(instruction)], len(dataconfig.instructions_map)
    )
    if isinstance(instruction, Circle):
        radius = quantize(instruction.r, dataconfig.max_radius)
        x = quantize(instruction.x, dataconfig.canvas_size)
        y = quantize(instruction.y, dataconfig.canvas_size)
        angle = quantize(instruction.angle, 360)
        return torch.cat(
            [
                instruction_type,
                x,
                y,
                radius,
                index_padding,
                index_padding,
                angle,
            ],
            dim=0,
        )
    elif isinstance(instruction, Square):
        size = quantize(instruction.size, dataconfig.max_radius)
        x = quantize(instruction.x, dataconfig.canvas_size)
        y = quantize(instruction.y, dataconfig.canvas_size)
        angle = quantize(instruction.angle, 360)

        return torch.cat(
            [instruction_type, x, y, size, index_padding, index_padding, angle],
            dim=0,
        )
    elif isinstance(instruction, Triangle):
        size = quantize(instruction.size, dataconfig.max_radius)
        x = quantize(instruction.x, dataconfig.canvas_size)
        y = quantize(instruction.y, dataconfig.canvas_size)
        angle = quantize(instruction.angle, 360)

        return torch.cat(
            [instruction_type, x, y, size, index_padding, index_padding, angle],
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
                to_onehot(instruction.index, dataconfig.num_primitives + 1),
                index_padding,
                parameter_padding,
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
                to_onehot(index1, dataconfig.num_primitives + 1),
                to_onehot(index2, dataconfig.num_primitives + 1),
                parameter_padding,
            ],
            dim=0,
        )

    elif isinstance(instruction, Rotate):
        angle = quantize(instruction.angle, dataconfig.canvas_size)
        index = to_onehot(instruction.index, dataconfig.num_primitives + 1)

        return torch.cat(
            [
                instruction_type,
                parameter_padding,
                parameter_padding,
                parameter_padding,
                index,
                index_padding,
                angle,
            ],
            dim=0,
        )

    else:
        raise Exception(f"Unknown instruction: {instruction}")
