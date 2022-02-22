from dataclasses import dataclass
from typing import Callable
from data.types import DataConfig
from models.types import Model


@dataclass(frozen=True)
class ProgramSynthesisTask:
    data_loader: Callable
    instructions_map: dict
    parse_input: Callable
    parse_program: Callable
    embed_input: Callable
    embed_program: Callable
    embedding_to_program: Callable
    scorer: Callable
    model: Model


dataconfig = DataConfig(
    canvas_size=100,
    dataset_size=3,
    min_radius=5,
    max_radius=20,
    num_sample_points=100,
    num_circles=2,
    instruction_embedding_size=26,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints
)
