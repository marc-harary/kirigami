from typing import Tuple
import torch
from torch import nn


__all__ = ["Symmetrize"]


class Symmetrize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt):
        out = []
        if isinstance(ipt, Tuple):
            out = tuple((((tens + torch.transpose(tens, -1, -2)) / 2) for tens in ipt))
        else:
            out = (ipt + torch.transpose(ipt, -1, -2)) / 2
        return out
