import numpy as np
import torch


def quantize(input: float, n: int) -> torch.Tensor:
    return torch.Tensor([input]).round() / n


def to_onehot(x: int, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.Tensor([x]).long(), n_classes).squeeze()


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def from_onehot(array: np.ndarray) -> int:
    return np.argmax(softmax(array))
