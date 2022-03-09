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
from embedding.types import ProgramStaticEmbeddings


@dataclass(frozen=True)
class ProgramSynthesisTask:
    data_loader: Callable
    parse_input: Callable
    parse_program: Callable
    embed_input: Callable
    program_embedding: ProgramStaticEmbeddings
    scorer: Callable
    model: Model
    visualize: Optional[Callable]
    dataset_name: str


dataconfig = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=5,
    max_radius=20,
    num_sample_points=100,
    num_primitives=2,
    random_primitives=False,
    num_modifiers=3,
    instruction_embedding_size=16,
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
    instruction_embedding_size=16,
    primitive_types=[Circle],
    modifier_types=[Translate],
    instructions_map=all_instructions,
    name="dataset1",
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
    instruction_embedding_size=20,
    primitive_types=[Square, Triangle, Circle],
    modifier_types=[Translate, Rotate, Constraint],
    instructions_map=all_instructions,
    name="random",
)

dataconfig_3 = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=3,
    max_radius=6,
    num_sample_points=50,
    num_primitives=3,
    random_primitives=False,
    num_modifiers=3,
    instruction_embedding_size=18,
    primitive_types=[Square, Circle, Triangle],
    modifier_types=[Translate, Constraint, Constraint],
    instructions_map=all_instructions,
    name="faces",
)

dataconfig_basic = DataConfig(
    canvas_size=100,
    dataset_size=1000,
    min_radius=3,
    max_radius=20,
    num_sample_points=50,
    num_primitives=2,
    random_primitives=False,
    num_modifiers=0,
    instruction_embedding_size=16,
    primitive_types=[Square, Circle],
    modifier_types=[],
    instructions_map=all_instructions,
    name="random",
)
