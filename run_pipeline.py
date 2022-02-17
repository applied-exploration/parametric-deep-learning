from models.neural import LightningNeuralNetModel
from models.pytorch.mlp import MultiLayerPerceptron
from data.data_loader import load_data
import torch.nn.functional as F
from embedding import embed_and_normalize_points, embed_instructions
from utils.parse import parse_points, parse_program
from generate_dataset import dataconfig


def run_pipeline():

    X, y = load_data()

    X = [parse_points(row) for row in X]
    y = [parse_program(row) for row in y]

    X = [embed_and_normalize_points(dataconfig, row) for row in X]
    y = [embed_instructions(dataconfig, row) for row in y]

    new_model = LightningNeuralNetModel(
        MultiLayerPerceptron(
            hidden_layers_ratio=[1.0],
            probabilities=False,
            loss_function=F.mse_loss,
        ),
        max_epochs=15,
    )

    new_model.fit(X, y)


if __name__ == "__main__":
    run_pipeline()
