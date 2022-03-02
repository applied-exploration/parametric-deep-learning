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
    all_instructions,
)
from models.types import Model


@dataclass(frozen=True)
class ProgramSynthesisTask:
    data_loader: Callable
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
    instruction_embedding_size=32,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints,
    primitive_types=[Circle],
    modifier_types=[Translate],
    instructions_map=all_instructions,
    name="test",

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
    instruction_embedding_size=28,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints,
    primitive_types=[Circle],
    modifier_types=[Translate],
    instructions_map=all_instructions,
    name="dataset1"
    
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
    instruction_embedding_size=28,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints
    primitive_types=[Square, Triangle, Circle],
    modifier_types=[Translate, Rotate, Constraint],
    instructions_map=all_instructions,
    name="random",
)

dataconfig_3 = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=9,
    max_radius=15,
    num_sample_points=50,
    num_primitives=3,
    random_primitives=False,
    num_modifiers=3,
    instruction_embedding_size=26,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints
    primitive_types=[Square, Triangle, Circle],
    modifier_types=[Translate, Constraint, Constraint],
    instructions_map=all_instructions,
    name="faces",
)
