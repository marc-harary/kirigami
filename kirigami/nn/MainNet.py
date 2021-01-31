from sys import modules
from typing import List
import torch
from torch import nn
import kirigami

__all__ = ['MainNet']

class MainNet(nn.Module):
    '''Constructs deep net from list of dictionaries'''
    def __init__(self, dict_list: List[dict]):
        super(MainNet, self).__init__()
        for i, layer_dict in enumerate(dict_list):
            obj_ptr = eval(layer_dict['class_name'])(**layer_dict['kwargs'])
            setattr(self, f'layer{i}', obj_ptr)
        self.n_layers = i

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        for i in range(self.n_layers + 1):
            layer = getattr(self, f'layer{i}')
            if isinstance(layer, nn.LSTM):
                out = torch.transpose(out, 1, 2)
                out, _ = layer(out)
                out = torch.transpose(out, 1, 2)
            else:
                out = layer(out)
        return out
