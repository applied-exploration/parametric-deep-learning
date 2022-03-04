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
from config import ProgramSynthesisTask, dataconfig_basic
from utils.scoring import score_programs
from render.visualize import visualize
from loss.compare_embeddings import compare_embedded_instructions_loss
import torch

dataconfig = dataconfig_basic

task = ProgramSynthesisTask(
    data_loader=load_data,
    parse_input=parse_grid,
    parse_program=parse_program,
    embed_input=embed_grid(dataconfig),
    embed_program=embed_instructions(dataconfig),
    embedding_to_program=from_embeddings_to_instructions(dataconfig),
    scorer=score_programs,
    model=LightningNeuralNetModel(
        ConvolutionalModel(
            loss_function=compare_embedded_instructions_loss(dataconfig),
        ),
        max_epochs=1,
    ),
    visualize=visualize(dataconfig),
    dataset_name=dataconfig.name,
)


def run_pipeline(task: ProgramSynthesisTask):

    X_train, y_train, X_test, y_test = load_data(task.dataset_name)

    X_train = [task.embed_input(task.parse_input(row)) for row in X_train]
    y_train = [task.embed_program(task.parse_program(row)) for row in y_train]
    X_test_without_embedding = [task.parse_input(row) for row in X_test]
    X_test = [task.embed_input(row) for row in X_test_without_embedding]
    y_test = [task.parse_program(row) for row in y_test]

    task.model.fit(X_train, y_train)
    output = task.model.predict(torch.stack(X_test, dim=0))
    output_programs = task.embedding_to_program(output)

    score = score_programs(y_test, output_programs)
    print(f"Score: {score}")
    if task.visualize is not None:
        task.visualize(X_test_without_embedding, output_programs)


if __name__ == "__main__":
    run_pipeline(task)
