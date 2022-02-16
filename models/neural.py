from __future__ import annotations
from models.base import Model
import numpy as np
import copy
import pytorch_lightning as pl

class LightningNeuralNetModel(Model):

    method = 'regression'
    data_transformation = 'transformed'
    only_column = None
    model_type = 'ml'

    ''' Standard lightning methods '''
    
    def __init__(self, model, max_epochs=5):
        self.model = model
        self.trainer = pl.Trainer(max_epochs=max_epochs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        train_dataloader = self.__prepare_data(X.astype(float), y.astype(float))
        self.trainer.fit(self.model, train_dataloader)
        
    def predict(self, X: np.ndarray) -> tuple[float, np.ndarray]:
        return self.model(X)

    def clone(self):
        model_copy = copy.deepcopy(self.model)
        return LightningNeuralNetModel(model_copy)
    
