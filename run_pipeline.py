from models.neural import LightningNeuralNetModel
from models.pytorch.mlp import MultiLayerPerceptron
from data.data_loader import load_data
import torch.nn.functional as F
from embedding import (
    embed_and_normalize_points,
    embed_instructions,
    from_embeddings_to_instructions,
    instructions,
)
from utils.parse import parse_points, parse_program
from config import dataconfig
from utils.scoring import score_instructions
from utils.visualize import display_features, display_program, display_both
from utils.render import render
from data.types import Circle, Translate, Constraint, DataConfig


def run_pipeline(display_plot: bool = False):

    X_train, y_train, X_test, y_test = load_data()

    X_train = [
        embed_and_normalize_points(dataconfig, parse_points(row)) for row in X_train
    ]
    y_train = [embed_instructions(dataconfig, parse_program(row)) for row in y_train]

    input_features_saved = [parse_points(row) for row in X_test]
    X_test = [
        embed_and_normalize_points(dataconfig, parse_points(row)) for row in X_test
    ]
    y_test = [parse_program(row) for row in y_test]

    model = LightningNeuralNetModel(
        MultiLayerPerceptron(
            hidden_layers_ratio=[1.0],
            probabilities=False,
            loss_function=F.mse_loss,
        ),
        max_epochs=150,
    )

    model.fit(X_train, y_train)
    output = model.predict(X_test)
    output_instructions = from_embeddings_to_instructions(output, dataconfig)

    score = score_instructions(output_instructions, y_test)
    print(f"Score: {score}")

    if display_plot:
        output_primitives = []
        for output_instruction in output_instructions:
            primitives, instructions = [], []
            for node in output_instruction:
                if type(node) == Circle:
                    primitives.append(node)
                else:
                    instructions.append(node)

            primitives, _ = render(primitives, instructions)
            output_primitives.append(primitives)

        display_both(
            input_features_saved,
            output_primitives,
            dataconfig,
        )


if __name__ == "__main__":
    run_pipeline(display_plot=True)
