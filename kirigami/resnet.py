from typing import *
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, ceil


class ResNetBlock(nn.Module):
    def __init__(self,
                 p: float,
                 dilations: Tuple[int,int],
                 kernel_sizes: Tuple[int],
                 n_channels: int,
                 act: str = "GELU",
                 **kwargs) -> None:
        super().__init__()
        self.resnet = True # resnet
        self.conv1 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[0],
                                     dilation=dilations[0],
                                     padding=self.get_padding(dilations[0], kernel_sizes[0]),
                                     bias=True)
        self.norm1 = torch.nn.InstanceNorm2d(n_channels)
        self.act1 = getattr(nn, act)()
        self.drop1 = torch.nn.Dropout(p=p)
        self.conv2 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[1],
                                     dilation=dilations[1],
                                     padding=self.get_padding(dilations[1], kernel_sizes[1]),
                                     bias=True)
        self.norm2 = nn.InstanceNorm2d(n_channels)
        self.act2 = getattr(nn, act)()


    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)
        if self.training:
            out = self.drop1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += ipt
        out = self.act2(out)
        return out

    @staticmethod
    def get_padding(dilation: int, kernel_size: int) -> int:
        return round((dilation * (kernel_size - 1)) / 2)


class ResNet(nn.Module):
    def __init__(self, n_blocks, n_channels, kernel_sizes, dilations, activation, dropout=0.5):
        super().__init__()
        if dilations is None:
            dilations = n_blocks * [1]
        else:
            num_cycles = ceil(2 * n_blocks / len(dilations))
            dilations = num_cycles * dilations
        trunk_list = [nn.Conv2d(9, n_channels, kernel_size=1, padding=0),
                      getattr(nn, activation)()]
        for i in range(n_blocks):
            block = ResNetBlock(p=dropout,
                                dilations=dilations[2*i:2*(i+1)],
                                kernel_sizes=kernel_sizes,
                                n_channels=n_channels,
                                act=activation)
            trunk_list.append(block)
        trunk_list.append(nn.Conv2d(n_channels, 1, kernel_size=1))
        trunk_list.append(nn.Sigmoid())
        self.trunk = nn.Sequential(*trunk_list)
    
    def forward(self, ipt):
        out = {}
        out["con"] = self.trunk(ipt) 
        out["dists"] = {}
        return out


class ResNetParallel(nn.Module):
    def __init__(self, n_trunk_blocks, n_trunk_channels,
                       n_con_blocks, n_con_channels,
                       dist_types, n_bins, n_dist_blocks, n_dist_channels,
                       kernel_sizes, dilations, activation, dropout):
        super().__init__()
        if dilations is None:
            dilations = n_blocks * [1]
        else:
            max_blocks = max(n_trunk_blocks, n_con_blocks, n_dist_blocks)
            num_cycles = ceil(2 * max_blocks / len(dilations))
            dilations = num_cycles * dilations
        trunk_list = [nn.Conv2d(9, n_trunk_channels, kernel_size=1, padding=0),
                      getattr(nn, activation)()]
        for i in range(n_trunk_blocks):
            block = ResNetBlock(p=dropout,
                                dilations=dilations[2*i:2*(i+1)],
                                kernel_sizes=kernel_sizes,
                                n_channels=n_trunk_channels,
                                act=activation)
            trunk_list.append(block)
        self.trunk = nn.Sequential(*trunk_list)

        con_list = [nn.Conv2d(n_trunk_channels, n_con_channels, kernel_size=1, padding=0)]
        for i in range(n_con_blocks):
            block = ResNetBlock(p=dropout,
                                dilations=dilations[2*i:2*(i+1)],
                                kernel_sizes=kernel_sizes,
                                n_channels=n_con_channels,
                                act=activation)
            con_list.append(block)
        con_list.append(nn.Conv2d(n_con_channels, 1, kernel_size=1))
        con_list.append(nn.Sigmoid())
        self.con_head = nn.Sequential(*con_list)

        self.dist_types = dist_types
        for dist_type in self.dist_types:
            track_list = [nn.Conv2d(n_trunk_channels, n_dist_channels, kernel_size=1, padding=0)]
            for j in range(n_dist_blocks):
                block = ResNetBlock(p=dropout,
                                    dilations=dilations[2*i:2*(i+1)],
                                    kernel_sizes=(3,5),
                                    n_channels=n_dist_channels,
                                    act=activation)
                track_list.append(block)
            track_list.append(nn.Conv2d(n_dist_channels, n_bins, kernel_size=1))
            track_list.append(nn.Softmax(-2))
            track = nn.Sequential(*track_list)
            setattr(self, f"{dist_type}_head", track)

    
    def forward(self, ipt):
        trunk_opt = self.trunk(ipt)
        out = {}
        out["con"] = self.con_head(trunk_opt)
        out["dists"] = {}
        for dist_type in self.dist_types:
            head = getattr(self, f"{dist_type}_head")
            out["dists"][dist_type] = head(trunk_opt)
        return out

