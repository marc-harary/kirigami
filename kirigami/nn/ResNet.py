'''implements sub-moduels for resnet'''

import torch
from torch import nn

__all__ = ['ActDropNorm']

class ActDropNorm(nn.Module):
    '''Performs activation, dropout, and batch normalization for resnet blocks'''
    def __init__(self, p: float, activation='ReLU', num_channels=8):
        super().__init__()
        activation_class = eval(activation)
        self.act = activation_class()
        self.drop = torch.nn.Dropout(p=p)
        self.norm = torch.nn.BatchNorm2d(num_channels)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act(out)
        out = self.drop(out)
        out = self.norm(out)
        return out
