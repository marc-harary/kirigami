'''loads and builds main model for training'''

from typing import List
import torch
from torch import nn
import kirigami

__all__ = ['MainNet']

class MainNet(nn.Module):
    '''Constructs deep net from list of dictionaries'''
    def __init__(self, layers: List[str]):
        super().__init__()
        i = 0
        for i, layer in enumerate(layers):
            setattr(self, f'layer{i}', eval(layer))
        self.n_layers = i

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        for i in range(self.n_layers+1):
            layer = getattr(self, f'layer{i}')
            if isinstance(layer, nn.LSTM):
                out = out.squeeze()
                out = out.T
                out, _ = layer(out)
                out = out.T
                out = out.unsqueeze(0)
            else:
                out = layer(out)
        return out
