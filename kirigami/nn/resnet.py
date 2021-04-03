'''implements sub-moduels for resnet'''


import torch
from torch import nn
from torch.nn import *


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


class ResNetBlock(nn.Module):
    '''Implements ResNet unit'''
    def __init__(self,
                 p: float,
                 activation='ReLU',
                 n_channels=8,
                 kernel_size1=3,
                 kernel_size2=5,
                 resnet=True):
        super().__init__()
        self.resnet = resnet
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size1,
                               padding=kernel_size1//2)
        self.conv2 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size2,
                               padding=kernel_size2//2)
        self.act_drop_norm = ActDropNorm(num_channels=n_channels,
                                         p=p,
                                         activation=activation)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act_drop_norm(out)
        out = self.conv1(out)
        out = self.act_drop_norm(out)
        out = self.conv2(out)
        if self.resnet:
            out += ipt
        return out



class ResNet(nn.Module):
    '''Implements ResNet'''
    def __init__(self,
                 n_blocks: int,
                 p: float,
                 activation='ReLU',
                 n_channels=8,
                 kernel_size1=3,
                 kernel_size2=5,
                 resnet=True):
        super().__init__()
        self.n_blocks = n_blocks
        for i in range(self.n_blocks):
            block =  ResNetBlock(p,
                                 activation,
                                 n_channels,
                                 kernel_size1,
                                 kernel_size2,
                                 resnet) 
            setattr(self, f'layer{i}', block)
    
    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        for i in range(self.n_blocks):
            out = getattr(self, f'layer{i}')(out)
        return out
