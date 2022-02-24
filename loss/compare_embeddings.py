from torch import Tensor
from typing import Callable
from data.types import DataConfig
from embedding.instructions import from_embeddings_to_instructions
from torch.nn import functional as F


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

        index_instructions_end = 3
        input_instruction_types = input_per_instruction[:, :, :index_instructions_end]
        target_instruction_types = target_per_instruction[:, :, :index_instructions_end]
        loss_instructions_types = F.cross_entropy(
            input_instruction_types, target_instruction_types
        )

        index_parameters_end = index_instructions_end + 3
        input_parameters = input_per_instruction[
            :, :, index_instructions_end:index_parameters_end
        ]
        target_parameters = target_per_instruction[
            :, :, index_instructions_end:index_parameters_end
        ]
        loss_parameters = F.mse_loss(input_parameters, target_parameters)

        index_index1_end = index_parameters_end + dataconfig.max_definition_len
        input_index1 = input_per_instruction[
            :, :, index_parameters_end:index_index1_end
        ]
        target_index1 = target_per_instruction[
            :, :, index_parameters_end:index_index1_end
        ]
        loss_index1 = F.mse_loss(input_index1, target_index1)

        index_index2_end = index_parameters_end + (dataconfig.max_definition_len * 2)
        input_index2 = input_per_instruction[:, :, index_index1_end:index_index2_end]
        target_index2 = target_per_instruction[:, :, index_index1_end:index_index2_end]
        loss_index2 = F.mse_loss(input_index2, target_index2)

        return loss_instructions_types + loss_parameters + loss_index1 + loss_index2
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
            tgt_commands[padding_mask.bool()].reshape(-1).long(),
        )
        loss_args = F.cross_entropy(
            args_logits[mask.bool()].reshape(-1, self.args_dim),
            tgt_args[mask.bool()].reshape(-1).long() + 1,
        )  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res

    return __compare_embedded_instructions_loss
