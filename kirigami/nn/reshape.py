from typing import Tuple
import torch
import torch.nn as nn


__all__ = ["Concatenate", "Squeeze", "Symmetrize", "Triu", "DropH0C0", "Unsqueeze"]


class Concatenate(nn.Module):
    """concatenates Bx4xL matrices into Bx8xLxL tensors"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        seq = ipt.unsqueeze(-1)
        # rows = torch.cat(seq.shape[-2] * [seq], -1)
        rows = seq.expand(-1, -1, -1, seq.shape[2])
        cols = rows.permute(0, 1, 3, 2)
        out = torch.cat((rows,cols), 1)
        return out


class Squeeze(nn.Module):
    """squeezes batch at one dimension for interfrace between convolutions and LSTM's"""
    
    squeeze_dim: int
    
    def __init__(self, squeeze_dim: int = 1):
        super().__init__()
        self.squeeze_dim = squeeze_dim

    def forward(self, ipt: torch.Tensor):
        return ipt.squeeze(self.squeeze_dim)


class Unsqueeze(nn.Module):
    """squeezes batch at one dimension for interfrace between convolutions and LSTM's"""
    
    squeeze_dim: int
    
    def __init__(self, squeeze_dim: int = 1):
        super().__init__()
        self.squeeze_dim = squeeze_dim

    def forward(self, ipt: torch.Tensor):
        return ipt.unsqueeze(self.squeeze_dim)


class Symmetrize(nn.Module):
    """symmetrizes square matrix"""

    def __init__(self):
        super().__init__()

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        return (ipt + ipt.permute(0, 1, 3, 2)) / 2


class Triu(nn.Module):
    """returns upper triangular matrix of input"""

    def __init__(self):
        super().__init__()

    def forward(self, ipt: torch.tensor) -> torch.tensor:
        return ipt.triu()


class DropH0C0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt: Tuple[torch.Tensor,Tuple[torch.Tensor,torch.Tensor]]) -> torch.Tensor:
        return ipt[0]
