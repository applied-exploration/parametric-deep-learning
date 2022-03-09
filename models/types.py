from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
from dataclasses import dataclass


class Model(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: list[torch.Tensor],
        y_train: list[torch.Tensor],
        X_val: list[torch.Tensor],
        y_val: list[torch.Tensor],
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: list[torch.Tensor]) -> np.ndarray:
        raise NotImplementedError
