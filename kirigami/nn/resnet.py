from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.nn import *


__all__ = ["ActDropNorm", "ResNetBlock", "ResNet"]


class ActDropNorm(nn.Module):
    """performs activation, dropout, and batch normalization for resnet blocks"""
    
    act: Module
    drop: Module
    norm: Module 

    def __init__(self,
                 p: float,
                 activation: str,
                 norm_type: str,
                 n_channels: int,
                 length: Optional[int]) -> None:
        super().__init__()
        activation_class = eval(activation)
        self.act = activation_class()
        self.drop = nn.Dropout2d(p=p)
        if norm_type == "LayerNorm":
            self.norm = nn.LayerNorm((n_channels, length, length))
        elif norm_type == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(n_channels)
        else:
            self.norm = nn.InstanceNorm2d(n_channels)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act(out)
        if self.training:
            out = self.drop(out)
            out = self.norm(out)
        return out


class ResNetBlock(nn.Module):
    """implements ResNet unit"""

    resnet: bool
    conv1: Module
    act_drop_norm1: Module
    conv2: Module
    act_drop_norm2: Module

    def __init__(self,
                 p: float,
                 dilations: Tuple[int,int],
                 kernel_sizes: Tuple[int],
                 activation: str,
                 norm_type: str,
                 length: Optional[int],
                 n_channels: int,
                 resnet: bool = True) -> None:
        super().__init__()
        self.resnet = resnet
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_sizes[0],
                               dilation=dilations[0],
                               padding=self.get_padding(dilations[0], kernel_sizes[0]))
        self.act_drop_norm1 = ActDropNorm(p=p,
                                          activation=activation,
                                          norm_type=norm_type,
                                          n_channels=n_channels,
                                          length=length)
        self.conv2 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_sizes[1],
                               dilation=dilations[1],
                               padding=self.get_padding(dilations[1], kernel_sizes[1]))
        self.act_drop_norm2 = ActDropNorm(p=p,
                                          activation=activation,
                                          norm_type=norm_type,
                                          n_channels=n_channels,
                                          length=length)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.conv1(out)
        out = self.act_drop_norm1(out)
        out = self.conv2(out)
        out = self.act_drop_norm2(out)
        if self.resnet:
            out += ipt
        return out

    @staticmethod
    def get_padding(dilation: int, kernel_size: int) -> int:
        """returns padding needed for 'same'-like feature in TensorFlow"""
        return round((dilation * (kernel_size - 1)) / 2)


class ResNet(nn.Module):
    """implements ResNet"""
    
    blocks: Sequential

    def __init__(self,
                 n_blocks: int,
                 p: float = 0.5,
                 dilations: Optional[List[int]] = None,
                 kernel_sizes: Tuple[int,int] = (3,5),
                 activation: str = "ReLU",
                 norm_type: str = "BatchNorm2d",
                 n_channels: int = 8,
                 length: Optional[int] = None,  
                 resnet: bool = True) -> None:
        super().__init__()
        block_list = []
        dilations = dilations or 2*n_blocks*[0]
        assert len(dilations) == 2*n_blocks, "Must pass in two dilations per block!"
        for i in range(n_blocks):
            block =  ResNetBlock(p=p,
                                 dilations=dilations[2*i:2*(i+1)],
                                 kernel_sizes=kernel_sizes,
                                 activation=activation,
                                 norm_type=norm_type,
                                 length=length,
                                 n_channels=n_channels,
                                 resnet=resnet) 
            block_list.append(block)
        self.blocks = Sequential(*block_list) 
    
    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        return self.blocks(ipt)
