import torch
import torch.nn as nn

__all__ = ["Triu"]

class Triu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt: torch.tensor) -> torch.tensor:
        return ipt.triu()
