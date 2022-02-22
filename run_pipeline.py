from models.neural import LightningNeuralNetModel
from models.pytorch.mlp import MultiLayerPerceptron
from data.data_loader import load_data
import torch.nn.functional as F
from embedding import (
    embed_and_normalize_points,
    embed_instructions,
    from_embeddings_to_instructions,
)
from utils.parse import parse_points, parse_program
from config import ProgramSynthesisTask, dataconfig
from utils.scoring import score_instructions


task = ProgramSynthesisTask(
    data_loader=load_data,
    instructions_map={"move": 0, "turn": 1, "draw": 2},
    parse_input=parse_points,
    parse_program=parse_program,
    embed_input=embed_and_normalize_points(dataconfig),
    embed_program=embed_instructions(dataconfig),
    embedding_to_program=from_embeddings_to_instructions(dataconfig),
    scorer=score_instructions,
    model=LightningNeuralNetModel(
        MultiLayerPerceptron(
            hidden_layers_ratio=[1.0],
            probabilities=False,
            loss_function=F.mse_loss,
        ),
        max_epochs=50,
    ),
)


def run_pipeline(task: ProgramSynthesisTask):

    X_train, y_train, X_test, y_test = load_data()

    X_train = [task.embed_input(task.parse_input(row)) for row in X_train]
    y_train = [task.embed_program(task.parse_program(row)) for row in y_train]

    X_test = [task.embed_input(task.parse_input(row)) for row in X_test]
    y_test = [task.parse_program(row) for row in y_test]

    task.model.fit(X_train, y_train)
    output = task.model.predict(X_test)
    output_instructions = task.embedding_to_program(output)

    score = score_instructions(output_instructions, y_test)
    print(f"Score: {score}")


if __name__ == "__main__":
    run_pipeline(task)
