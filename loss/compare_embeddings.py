from torch import Tensor
from typing import Callable
from data.types import DataConfig
from torch.nn import functional as F
import torch


def compare_embedded_instructions_loss(dataconfig: DataConfig) -> Callable:
    def __compare_embedded_instructions_loss(
        input: Tensor,
        target: Tensor,
    ) -> Tensor:

        assert input.shape[1] % dataconfig.instruction_embedding_size == 0
        assert input.shape[0] == target.shape[0]

        no_of_instructions = int(input.shape[1] / dataconfig.instruction_embedding_size)
        input_per_instruction = input.view(
            input.shape[0], no_of_instructions, dataconfig.instruction_embedding_size
        )
        target_per_instruction = target.view(
            target.shape[0], no_of_instructions, dataconfig.instruction_embedding_size
        )

        index_instructions_end = len(dataconfig.instructions_map)
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

        index_index1_end = index_parameters_end + dataconfig.max_definition_len
        input_index1 = input_per_instruction[
            :, :, index_parameters_end:index_index1_end
        ]
        target_index1 = target_per_instruction[
            :, :, index_parameters_end:index_index1_end
        ]

        loss_index1 = F.cross_entropy(
            input_index1.view(-1, dataconfig.max_definition_len),
            torch.argmax(target_index1.view(-1, dataconfig.max_definition_len), dim=1),
            ignore_index=0,
        )

        index_index2_end = index_parameters_end + (dataconfig.max_definition_len * 2)
        input_index2 = input_per_instruction[:, :, index_index1_end:index_index2_end]
        target_index2 = target_per_instruction[:, :, index_index1_end:index_index2_end]
        loss_index2 = F.cross_entropy(
            input_index2.view(-1, dataconfig.max_definition_len),
            torch.argmax(target_index2.view(-1, dataconfig.max_definition_len), dim=1),
            ignore_index=0,
        )

        index_angle_end = index_index2_end + 1
        input_angle = input_per_instruction[:, :, index_index2_end:index_angle_end]
        target_angle = target_per_instruction[:, :, index_index2_end:index_angle_end]
        padding_mask = (target_angle != -1.0).int()
        input_angle = input_angle * padding_mask
        target_angle = target_angle * padding_mask
        loss_angle = F.mse_loss(input_angle, target_angle) * 50

        return loss_instructions_types + loss_parameters + loss_index1 + loss_index2 + loss_angle

    return __compare_embedded_instructions_loss
