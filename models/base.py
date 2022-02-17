from __future__ import annotations
from typing import Literal, Optional, Union
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError
    
