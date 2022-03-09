from models.neural import LightningNeuralNetModel
from models.pytorch.cnn import ConvolutionalModel
from data.data_loader import load_data
from embedding import embed_grid
from embedding.instructions_mixed import MixedProgramStaticEmbeddings
from utils.parse import parse_grid, parse_program
from config import ProgramSynthesisTask
from utils.scoring import score_programs
from render.visualize import visualize
from run_pipeline import run_pipeline
from data.generator import generator
from dataclasses import dataclass

from data.types import (
    DataConfig,
    Circle,
    Square,
    all_instructions,
)

dataconfig = DataConfig(
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
    name="test_1",
)
program_embedding = MixedProgramStaticEmbeddings(dataconfig)

generator(config=dataconfig, display_plot=False)

task = ProgramSynthesisTask(
    data_loader=load_data,
    parse_input=parse_grid(dataconfig),
    parse_program=parse_program,
    embed_input=embed_grid(dataconfig),
    program_embedding=program_embedding,
    scorer=score_programs,
    model=LightningNeuralNetModel(
        ConvolutionalModel(loss_function=program_embedding.loss, dropout_p=0.2),
        max_epochs=100,
        logging=False,
    ),
    visualize=None,
    dataset_name=dataconfig.name,
)

run_pipeline(task)
