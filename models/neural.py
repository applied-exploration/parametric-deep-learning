from __future__ import annotations
from models.types import Model
import numpy as np
import pytorch_lightning as pl
from .pytorch.dataset import get_dataloader
import torch


class LightningNeuralNetModel(Model):
    def __init__(self, model, max_epochs=5):
        self.model = model
        self.trainer = pl.Trainer(max_epochs=max_epochs)

    def fit(self, X: list[torch.Tensor], y: list[torch.Tensor]) -> None:
        dataloader = get_dataloader(X, y)
        self.model.initialize_network(X[0].shape, y[0].shape[0])
        self.trainer.fit(self.model, dataloader)

    def predict(self, X: list[torch.Tensor]) -> np.ndarray:
        return self.model(X)
