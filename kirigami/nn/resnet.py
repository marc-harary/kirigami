from typing import Tuple
import torch
from torch import nn
from torch.nn import *


__all__ = ["ActDropNorm"]


class ActDropNorm(nn.Module):
    """Performs activation, dropout, and batch normalization for resnet blocks"""
    def __init__(self,
                 p: float,
                 activation: str,
                 norm_type: str,
                 shape: Tuple[int,int,int]) -> None:
        super().__init__()
        activation_class = eval(activation)
        self.act = activation_class()
        self.drop = torch.nn.Dropout2d(p=p)
        if norm_type == "LayerNorm":
            self.norm = torch.nn.LayerNorm(shape)
        elif norm_type == "BatchNorm2d":
            self.norm = torch.nn.BatchNorm2d(shape[0])

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act(out)
        if self.training:
            out = self.drop(out)
            out = self.norm(out)
        return out


class ResNetBlock(nn.Module):
    """Implements ResNet unit"""

    resnet: bool
    conv1: Module
    act_drop_norm1: Module
    conv2: Module
    act_drop_norm2: Module

    def __init__(self,
                 p: float,
                 activation: str = "ReLU",
                 norm_type: str = "LayerNorm",
                 shape: Tuple[int,int,int] = (8, 512, 512), 
                 kernel_size1: int = 3,
                 kernel_size2: int = 5,
                 resnet: bool = True):
        super().__init__()
        self.resnet = resnet
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size1,
                               padding=kernel_size1//2)
        self.act_drop_norm1 = ActDropNorm(p=p,
                                          activation=activation,
                                          norm_type=norm_type,
                                          shape=shape)
        self.conv2 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size2,
                               padding=kernel_size2//2)
        self.act_drop_norm2 = ActDropNorm(p=p,
                                          activation=activation,
                                          norm_type=norm_type,
                                          shape=shape)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.conv1(out)
        out = self.act_drop_norm1(out)
        out = self.conv2(out)
        out = self.act_drop_norm2(out)
        if self.resnet:
            out += ipt
        return out


class ResNet(nn.Module):
    """Implements ResNet"""
    
    blocks: Sequential

    def __init__(self,
                 n_blocks: int,
                 p: float,
                 activation: str = "ReLU",
                 norm_type: str = "LayerNorm",
                 shape: Tuple[int,int,int] = (8, 512, 512), 
                 kernel_size1: int = 3,
                 kernel_size2: int = 5,
                 resnet: bool = True) -> None:
        super().__init__()
        block_list = []
        for i in range(self.n_blocks):
            block =  ResNetBlock(p=p,
                                 activation=activation,
                                 norm_type=norm_type,
                                 shape=shape,
                                 kernel_size1=kernel_size1,
                                 kernel_size2=kernel_size2,
                                 resnet=resnet) 
            block_list.append(block)
        self.blocks = Sequential(*block_list) 
    
    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        return self.blocks(ipt)
