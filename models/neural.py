from __future__ import annotations
from models.base import Model
import numpy as np
import pytorch_lightning as pl


class LightningNeuralNetModel(Model):

    """Standard lightning methods"""

    def __init__(self, model, max_epochs=5):
        self.model = model
        self.trainer = pl.Trainer(max_epochs=max_epochs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        train_dataloader = self.__prepare_data(X.astype(float), y.astype(float))
        self.model.initialize_network(X.shape[1], y.shape[1])
        self.trainer.fit(self.model, train_dataloader)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model(X)
