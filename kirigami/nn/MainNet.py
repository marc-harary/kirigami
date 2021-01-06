import sys
sys.path.append('../utils')

import torch.nn as nn
import torch.nn.functional as F

from constants import *


class MainNet(nn.Module):
    def __init__(self, dict_list):
        super(MainNet, self).__init__()
        for i, layer_dict in enumerate(dict_list):
            layer_func = layer_dict['layer']
            layer_class = getattr(nn, layer_func)
            layer_obj = layer_class(**layer_dict['params'])
            setattr(self, f'layer{i}', layer_obj)
        self.n_layers = i
    
    def forward(self, input):
        ret = input
        for i in range(self.n_layers + 1):
            layer = getattr(self, f'layer{i}')
            ret = layer(ret)
        return ret 
