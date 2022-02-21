import numpy as np
import torch


def quantize(input: torch.Tensor, n=256) -> torch.Tensor:
    return input.round().clip(min=0, max=n - 1)


def to_onehot(x: int, n_classes: int) -> torch.Tensor:
    x = np.array([x])
    return torch.Tensor(np.identity(n_classes)[x]).squeeze()


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def from_onehot(array: np.ndarray) -> int:
    return np.argmax(softmax(array))
