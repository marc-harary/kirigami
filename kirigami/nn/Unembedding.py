import re
import torch
from torch import nn


__all__ =['AbstractUnembedding', 'BpseqUnembedding']


class AbstractUnembedding(nn.Module):
    def __init__(self):
        super(AbstractUnembedding, self).__init__()


class BpseqUnembedding(AbstractUnembedding):
    def __init__(self):
        super(AbstractUnembedding, self).__init__()

    def forward(self, sequence: str, label: torch.Tensor) -> str:
        assert label.dim == 4 and label.size[:2] == (1,1)
        lines = []
        for char, (i, row) in zip(sequence, enumerate(label)):
            j = row.nonzero().item()
            line = f'{i+1} {char} {j+1}'
            lines.append(line)
        return lines
