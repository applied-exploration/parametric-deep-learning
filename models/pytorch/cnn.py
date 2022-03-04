import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
import torch
from typing import Callable
import functools
import operator


class ConvolutionalModel(pl.LightningModule):
    def __init__(self, loss_function: Callable, learning_rate=2e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def forward(self, x):
        x = self.feature_extractor(x.unsqueeze(dim=1))
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output 

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def initialize_network(self, input_dim: tuple[int, int], output_dim: int) -> None:
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(
                self.feature_extractor(torch.rand(1, *input_dim).unsqueeze(dim=1)).shape
            ),
        )

        self.out = nn.Linear(num_features_before_fcnn, output_dim)
