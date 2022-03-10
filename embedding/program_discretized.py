import torch
from data.types import (
    Circle,
    Constraint,
    Program,
    Translate,
    Triangle,
    Square,
    Rotate,
    Instruction,
)
from generate_dataset import DataConfig
from .utils import from_onehot, to_onehot
import numpy as np
from .types import ProgramStaticEmbeddings
from torch import Tensor
from torch.nn import functional as F


class DiscretizedProgramStaticEmbeddings(ProgramStaticEmbeddings):
    def __init__(self, dataconfig: DataConfig) -> None:
        self.dataconfig = dataconfig

        # we always add one to the available classes, for padding

        self.tnsr_idx_instruction_type_end = len(self.dataconfig.instructions_map)
        self.tnsr_idx_parameter_1_end = (
            self.tnsr_idx_instruction_type_end + self.dataconfig.canvas_size + 1
        )
        self.tnsr_idx_parameter_2_end = (
            self.tnsr_idx_parameter_1_end + self.dataconfig.canvas_size + 1
        )
        self.tnsr_idx_parameter_3_end = (
            self.tnsr_idx_parameter_2_end + self.dataconfig.canvas_size + 1
        )
        self.tnsr_idx_index_1_end = (
            self.tnsr_idx_parameter_3_end + self.dataconfig.num_primitives + 1
        )
        self.tnsr_idx_index_2_end = (
            self.tnsr_idx_index_1_end + self.dataconfig.num_primitives + 1
        )
        self.tnsr_idx_angle_end = self.tnsr_idx_index_2_end + 361

        self.single_instruction_size = self.tnsr_idx_angle_end

    def program_to_tensor(self, program: Program) -> torch.Tensor:
        """Embed an instruction into a tensor.
        The first two dimensions are one-hot encoded versions of the type.
        The last dimensions are embedded parameters (padded if not present).
        """
        to_flatten = [
            embed_instruction(self.dataconfig, instruction) for instruction in program
        ]
        return torch.flatten(
            torch.stack(
                to_flatten,
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
            instruction_type = from_onehot(
                embedding[: self.tnsr_idx_instruction_type_end]
            )
            parameter_1 = from_onehot(
                embedding[no_of_instruction_types : self.tnsr_idx_parameter_1_end]
            )
            parameter_2 = from_onehot(
                embedding[self.tnsr_idx_parameter_1_end : self.tnsr_idx_parameter_2_end]
            )
            parameter_3 = from_onehot(
                embedding[self.tnsr_idx_parameter_2_end : self.tnsr_idx_parameter_3_end]
            )
            index_1 = from_onehot(
                embedding[self.tnsr_idx_parameter_3_end : self.tnsr_idx_index_1_end]
            )
            index_2 = from_onehot(
                embedding[self.tnsr_idx_index_1_end : self.tnsr_idx_index_2_end]
            )
            angle = from_onehot(
                embedding[self.tnsr_idx_index_2_end : self.tnsr_idx_angle_end]
            )

            if instruction_type == self.dataconfig.instructions_map[Circle]:
                return Circle(
                    x=parameter_1,
                    y=parameter_2,
                    r=parameter_3,
                    angle=angle,
                )
            elif instruction_type == self.dataconfig.instructions_map[Square]:
                return Square(
                    x=parameter_1,
                    y=parameter_2,
                    size=parameter_3,
                    angle=angle,
                )
            elif instruction_type == self.dataconfig.instructions_map[Triangle]:
                return Triangle(
                    x=parameter_1,
                    y=parameter_2,
                    size=parameter_3,
                    angle=angle,
                )
            elif instruction_type == self.dataconfig.instructions_map[Translate]:
                return Translate(
                    x=parameter_1,
                    y=parameter_2,
                    index=index_1,
                )
            elif instruction_type == self.dataconfig.instructions_map[Constraint]:
                return Constraint(
                    x=parameter_1,
                    y=parameter_2,
                    indicies=(index_1, index_2),
                )
            elif instruction_type == self.dataconfig.instructions_map[Rotate]:
                return Rotate(
                    index=index_1,
                    angle=angle,
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

        input_instruction_types = input_per_instruction[
            :, :, : self.tnsr_idx_instruction_type_end
        ]
        target_instruction_types = target_per_instruction[
            :, :, : self.tnsr_idx_instruction_type_end
        ]
        loss_instructions_types = F.cross_entropy(
            input_instruction_types,
            target_instruction_types,
        )

        input_parameter_1 = input_per_instruction[
            :, :, self.tnsr_idx_instruction_type_end : self.tnsr_idx_parameter_1_end
        ]
        target_parameter_1 = target_per_instruction[
            :, :, self.tnsr_idx_instruction_type_end : self.tnsr_idx_parameter_1_end
        ]
        loss_parameter_1 = F.cross_entropy(
            input_parameter_1,
            target_parameter_1,
        )

        input_parameter_2 = input_per_instruction[
            :, :, self.tnsr_idx_parameter_1_end : self.tnsr_idx_parameter_2_end
        ]
        target_parameter_2 = target_per_instruction[
            :, :, self.tnsr_idx_parameter_1_end : self.tnsr_idx_parameter_2_end
        ]
        loss_parameter_2 = F.cross_entropy(
            input_parameter_2,
            target_parameter_2,
        )

        input_parameter_3 = input_per_instruction[
            :, :, self.tnsr_idx_parameter_2_end : self.tnsr_idx_parameter_3_end
        ]
        target_parameter_3 = target_per_instruction[
            :, :, self.tnsr_idx_parameter_2_end : self.tnsr_idx_parameter_3_end
        ]
        loss_parameter_3 = F.cross_entropy(
            input_parameter_3,
            target_parameter_3,
        )

        input_index1 = input_per_instruction[
            :, :, self.tnsr_idx_parameter_3_end : self.tnsr_idx_index_1_end
        ]
        target_index1 = target_per_instruction[
            :, :, self.tnsr_idx_parameter_3_end : self.tnsr_idx_index_1_end
        ]
        loss_index1 = F.cross_entropy(
            input_index1,
            target_index1,
        )

        input_index2 = input_per_instruction[
            :, :, self.tnsr_idx_index_1_end : self.tnsr_idx_index_2_end
        ]
        target_index2 = target_per_instruction[
            :, :, self.tnsr_idx_index_1_end : self.tnsr_idx_index_2_end
        ]
        loss_index2 = F.cross_entropy(
            input_index2,
            target_index2,
        )

        input_angle = input_per_instruction[
            :, :, self.tnsr_idx_index_2_end : self.tnsr_idx_angle_end
        ]
        target_angle = target_per_instruction[
            :, :, self.tnsr_idx_index_2_end : self.tnsr_idx_angle_end
        ]
        loss_angle = F.cross_entropy(input_angle, target_angle)

        return (
            loss_instructions_types
            + loss_parameter_1
            + loss_parameter_2
            + loss_parameter_3
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
    parameter_padding = to_onehot(dataconfig.canvas_size, dataconfig.canvas_size + 1)
    index_padding = to_onehot(0, dataconfig.num_primitives + 1)
    angle_padding = to_onehot(360, 361)
    instruction_type = to_onehot(
        dataconfig.instructions_map[type(instruction)], len(dataconfig.instructions_map)
    )
    if isinstance(instruction, Circle):
        radius = to_onehot(int(instruction.r), dataconfig.canvas_size + 1)
        x = to_onehot(int(instruction.x), dataconfig.canvas_size + 1)
        y = to_onehot(int(instruction.y), dataconfig.canvas_size + 1)
        angle = to_onehot(int(instruction.angle), 361)
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
        size = to_onehot(int(instruction.size), dataconfig.canvas_size + 1)
        x = to_onehot(int(instruction.x), dataconfig.canvas_size + 1)
        y = to_onehot(int(instruction.y), dataconfig.canvas_size + 1)
        angle = to_onehot(int(instruction.angle), 361)

        return torch.cat(
            [instruction_type, x, y, size, index_padding, index_padding, angle],
            dim=0,
        )
    elif isinstance(instruction, Triangle):
        size = to_onehot(int(instruction.size), dataconfig.canvas_size + 1)
        x = to_onehot(int(instruction.x), dataconfig.canvas_size + 1)
        y = to_onehot(int(instruction.y), dataconfig.canvas_size + 1)
        angle = to_onehot(int(instruction.angle), 361)

        return torch.cat(
            [instruction_type, x, y, size, index_padding, index_padding, angle],
            dim=0,
        )
    elif isinstance(instruction, Translate):
        x = to_onehot(int(instruction.x), dataconfig.canvas_size + 1)
        y = to_onehot(int(instruction.y), dataconfig.canvas_size + 1)
        return torch.cat(
            [
                instruction_type,
                x,
                y,
                parameter_padding,
                to_onehot(instruction.index, dataconfig.num_primitives + 1),
                index_padding,
                angle_padding,
            ],
            dim=0,
        )

    elif isinstance(instruction, Constraint):
        x = to_onehot(int(instruction.x), dataconfig.canvas_size + 1)
        y = to_onehot(int(instruction.y), dataconfig.canvas_size + 1)
        index1, index2 = instruction.indicies
        return torch.cat(
            [
                instruction_type,
                x,
                y,
                parameter_padding,
                to_onehot(index1, dataconfig.num_primitives + 1),
                to_onehot(index2, dataconfig.num_primitives + 1),
                angle_padding,
            ],
            dim=0,
        )

    elif isinstance(instruction, Rotate):
        angle = to_onehot(int(instruction.angle), 361)
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
