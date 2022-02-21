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
from config import dataconfig
from utils.scoring import score_instructions


def run_pipeline():

    X_train, y_train, X_test, y_test = load_data()

    X_train = [
        embed_and_normalize_points(dataconfig, parse_points(row)) for row in X_train
    ]
    y_train = [embed_instructions(dataconfig, parse_program(row)) for row in y_train]

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
        max_epochs=50,
    )

    model.fit(X_train, y_train)
    output = model.predict(X_test)
    output_instructions = from_embeddings_to_instructions(output, dataconfig)

    score = score_instructions(output_instructions, y_test)
    print(f"Score: {score}")


if __name__ == "__main__":
    run_pipeline()
