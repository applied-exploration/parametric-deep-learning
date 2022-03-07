from __future__ import annotations
from models.types import Model
import numpy as np
import pytorch_lightning as pl
from .pytorch.dataset import get_dataloader
import torch


class LightningNeuralNetModel(Model):
    def __init__(self, model: pl.LightningModule, max_epochs: int, logging: bool):
        self.model = model

        if logging == True:
            from wandb_setup import get_wandb
            get_wandb()
            
        wandb_logger = (
            pl.loggers.WandbLogger(project="Parametric-Deep-Learning", log_model=True)
            if logging
            else None
        )
        self.trainer = pl.Trainer(
            logger=wandb_logger, max_epochs=max_epochs, check_val_every_n_epoch=5
        )

    def fit(
        self,
        X_train: list[torch.Tensor],
        y_train: list[torch.Tensor],
        X_val: list[torch.Tensor],
        y_val: list[torch.Tensor],
    ) -> None:
        train_dataloader = get_dataloader(X_train, y_train)
        val_dataloader = get_dataloader(X_val, y_val, shuffle=False)

        self.model.initialize_network(X_train[0].shape, y_train[0].shape[0])
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def predict(self, X: list[torch.Tensor]) -> np.ndarray:
        return self.model(X)
