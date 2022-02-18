from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch


class Dataset(Dataset):
    def __init__(self, X: list[torch.Tensor], y: list[torch.Tensor]):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(
    X: list[torch.Tensor],
    y: list[torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True,
):
    training_data = Dataset(X, y)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader
