import torch
from torch import nn

__all__ = ["Symmetrizer"]

class Symmetrizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        return (ipt + ipt.permute(0, 1, 3, 2)) / 2
