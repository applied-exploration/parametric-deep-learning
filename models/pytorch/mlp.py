import torch
from torch import nn
import pytorch_lightning as pl
import math
import torch
from typing import Callable


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(
        self,
        hidden_layers_ratio: list[float],
        dropout_ratio: float,
        loss_function: Callable,
    ):
        super().__init__()
        self.hidden_layers_ratio = hidden_layers_ratio
        self.loss_function = loss_function
        self.dropout_ratio = dropout_ratio

    def initialize_network(self, input_dim: tuple[int, int], output_dim: int) -> None:
        self.layers = nn.ModuleList()
        current_dim = input_dim[0]

        for hdim in self.hidden_layers_ratio:
            hidden_layer_size = int(math.floor(current_dim * hdim))
            self.layers.append(nn.Linear(current_dim, hidden_layer_size))
            self.layers.append(nn.Dropout(p=self.dropout_ratio))
            self.layers.append(nn.ReLU())
            current_dim = hidden_layer_size

        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: list[torch.Tensor]):
        # in lightning, forward defines the prediction/inference actions
        x = torch.stack(x, dim=0)

        for layer in self.layers:
            x = layer(x)

        return x

    def training_step(self, batch: torch.Tensor, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)

        loss = 0
        for layer in self.layers:
            x = layer(x.float())

        loss = self.loss_function(x, y.float())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
