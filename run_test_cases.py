from models.neural import LightningNeuralNetModel
from models.pytorch.cnn import ConvolutionalModel
from data.data_loader import load_data
import torch.nn.functional as F
from embedding import (
    embed_grid,
    embed_instructions,
    from_embeddings_to_instructions,
)
from utils.parse import parse_grid, parse_program
from config import ProgramSynthesisTask
from utils.scoring import score_programs
from render.visualize import visualize
from loss.compare_embeddings import compare_embedded_instructions_loss
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

generator(config=dataconfig, display_plot=False)

task = ProgramSynthesisTask(
    data_loader=load_data,
    parse_input=parse_grid(dataconfig),
    parse_program=parse_program,
    embed_input=embed_grid(dataconfig),
    embed_program=embed_instructions(dataconfig),
    embedding_to_program=from_embeddings_to_instructions(dataconfig),
    scorer=score_programs,
    model=LightningNeuralNetModel(
        ConvolutionalModel(
            loss_function=compare_embedded_instructions_loss(dataconfig), dropout_p=0.2
        ),
        max_epochs=100,
        logging=False,
    ),
    visualize=None,
    dataset_name=dataconfig.name,
)

run_pipeline(task)
