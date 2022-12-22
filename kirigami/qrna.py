from typing import *
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log


class QRNABlock(nn.Module):
    def __init__(self,
                 p: float,
                 dilations: Tuple[int,int],
                 kernel_sizes: Tuple[int],
                 n_channels: int,
                 resnet: bool,
                 norm: str,
                 **kwargs) -> None:
        super().__init__()
        self.resnet = resnet
        self.conv1 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[0],
                                     dilation=dilations[0],
                                     padding=self.get_padding(dilations[0], kernel_sizes[0]))
        # self.norm1 = torch.nn.InstanceNorm2d(n_channels)
        self.norm1 = getattr(nn, norm)(n_channels)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=p)
        # self.relu1 = torch.nn.ReLU(inplace=True)
        # self.drop1 = torch.nn.Dropout(p=p, inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[1],
                                     dilation=dilations[1],
                                     padding=self.get_padding(dilations[1], kernel_sizes[1]))
        self.norm2 = getattr(nn, norm)(n_channels)
        # self.norm2 = torch.nn.BatchNorm2d(n_channels)
        self.relu2 = torch.nn.ReLU()


    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        if self.training:
            out = self.drop1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += ipt
        out = self.relu2(out)
        return out

    @staticmethod
    def get_padding(dilation: int, kernel_size: int) -> int:
        return round((dilation * (kernel_size - 1)) / 2)

