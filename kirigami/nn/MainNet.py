'''loads and builds main model for training'''

from typing import List
import torch
from torch import nn
import kirigami

__all__ = ['MainNet']

class MainNet(nn.Module):
    '''Constructs deep net from list of dictionaries'''
    def __init__(self, dict_list: List[dict]):
        super().__init__()
        i = 0
        for layer_dict in dict_list:
            obj_ptr = eval(layer_dict['class_name'])(**layer_dict['kwargs'])
            setattr(self, f'layer{i}', obj_ptr)
            i += 1
        self.n_layers = i

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        for i in range(self.n_layers + 1):
            layer = getattr(self, f'layer{i}')
            if isinstance(layer, nn.LSTM):
                out = torch.transpose(out, 1, 2)
                out, _ = layer(out)
                out = torch.transpose(out, 1, 2)
            else:
                out = layer(out)
        return out
