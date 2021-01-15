from sys import modules
import torch
from torch import nn
from . import *

class MainNet(nn.Module):
    def __init__(self, dict_list):
        super(MainNet, self).__init__()
        for i, layer_dict in enumerate(dict_list):
            layer_module = modules[layer_dict['module']]
            layer_class = getattr(layer_module, layer_dict['class'])
            layer_obj = layer_class(**layer_dict['kwargs'])
            setattr(self, f'layer{i}', layer_obj)
        self.n_layers = i

    def forward(self, input):
        ret = input
        for i in range(self.n_layers + 1):
            layer = getattr(self, f'layer{i}')
            if isinstance(layer, nn.LSTM):
                ret = torch.transpose(ret, 1, 2)
                ret, _ = layer(ret)
                ret = torch.transpose(ret, 1, 2)
            else:
                ret = layer(ret)
        return ret
