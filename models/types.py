from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch

class Model(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: list[torch.Tensor],
        y_train: list[torch.Tensor],
        X_val: list[torch.Tensor],
        y_val: list[torch.Tensor],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: list[torch.Tensor]) -> np.ndarray:
        raise NotImplementedError
