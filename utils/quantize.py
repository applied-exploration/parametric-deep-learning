import torch

def quantize(input: torch.Tensor, n=256) -> torch.Tensor:
    return input.round().clip(min=0, max=n - 1)
