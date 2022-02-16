from models.neural import LightningNeuralNetModel
from models.pytorch.mlp import MultiLayerPerceptron


def run_pipeline():
    new_model = LightningNeuralNetModel(
            MultiLayerPerceptron(
                hidden_layers_ratio = [1.0], 
                probabilities = False, 
                loss_function = F.mse_loss), 
            max_epochs=15
        )