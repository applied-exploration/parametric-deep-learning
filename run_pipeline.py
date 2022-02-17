from models.neural import LightningNeuralNetModel
from models.pytorch.mlp import MultiLayerPerceptron
from data.data_loader import load_data
import torch
import torch.nn.functional as F


def run_pipeline():

    X, y = load_data()
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
