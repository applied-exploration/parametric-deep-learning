from abc import ABC
from data.types import Program
from torch import Tensor
from data.types import DataConfig

class ProgramStaticEmbeddings(ABC):
    def __init__(self, dataconfig: DataConfig) -> None:
        raise NotImplementedError()

    def program_to_tensor(self, program: Program) -> Tensor:
        raise NotImplementedError()

    def tensor_to_programs(self, tensor: Tensor) -> list[Program]:
        raise NotImplementedError()

    def get_single_instruction_size(self) -> int:
        raise NotImplementedError()

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()
