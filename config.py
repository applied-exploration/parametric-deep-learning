from dataclasses import dataclass
from typing import Callable, Optional

from data.types import (
    DataConfig,
    Circle,
    Square,
    Triangle,
    Translate,
    Rotate,
    Constraint,
    Instruction,
)
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
    visualize: Optional[Callable]


dataconfig = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=5,
    max_radius=20,
    num_sample_points=100,
    num_primitives=2,
    random_primitives=False,
    num_modifiers=3,
    instruction_embedding_size=30,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints,
    primitive_types=[Circle],
    modifier_types=[Translate],
)
dataconfig_1 = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=5,
    max_radius=20,
    num_sample_points=100,
    num_primitives=2,
    random_primitives=False,
    num_modifiers=3,
    instruction_embedding_size=26,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints,
    primitive_types=[Circle],
    modifier_types=[Translate],
)
dataconfig_2 = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=9,
    max_radius=20,
    num_sample_points=50,
    num_primitives=4,
    random_primitives=False,
    num_modifiers=5,
    instruction_embedding_size=26,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints
    primitive_types=[Square, Triangle, Circle],
    modifier_types=[Translate, Rotate, Constraint],
)
