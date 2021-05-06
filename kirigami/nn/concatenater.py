import torch
import torch.nn as nn

__all__ = ["Concatenater"]

class Concatenater(nn.Module):
    """concatenates Bx4xL matrices into Bx8xLxL tensors"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, ipt: torch.tensor) -> torch.tensor:
        seq = ipt.unsqueeze(-1)
        # rows = torch.cat(seq.shape[-2] * [seq], -1)
        rows = seq.expand(-1, -1, -1, seq.shape[2])
        cols = rows.permute(0, 1, 3, 2)
        out = torch.cat((rows,cols), 1)
        return out
