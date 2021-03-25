import torch
from torch import nn

__all__ = ["Symmetrize"]

class Symmetrize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        ipt += ipt.permute(0, 1, 3, 2)
        return ipt / 2
