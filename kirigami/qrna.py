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



class QRNANet(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 n_bins: int = 38,
                 include_bin: bool = False,
                 include_inv: bool = False,
                 n_dists: int = 10,
                 n_channels: int = 32,
                 p: float = .2):

        super().__init__()

        self.n_blocks = n_blocks
        self.n_dists = n_dists
        self.n_bins = n_bins
        self.n_channels = n_channels

        self.in_channels = 9
        self.kernel_sizes = (3, 5)
        self.dilations = 2 * self.n_blocks * [1]
        self.fork_kernel_size = 5
        self.fork_padding = (self.fork_kernel_size - 1) // 2

        self.conv_init = torch.nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=n_channels,
                                         kernel_size=self.kernel_sizes[0],
                                         padding=1)

        for i in range(self.n_blocks):
            block = QRNABlock(p=p,
                              dilations=self.dilations[2*i:2*(i+1)],
                              kernel_sizes=self.kernel_sizes,
                              n_channels=n_channels,
                              resnet=True)
            setattr(self, f"block{i}", block)

        # self.con_conv = nn.Conv2d(in_channels=self.n_channels,
        #                           out_channels=1,
        #                           kernel_size=self.fork_kernel_size,
        #                           padding=self.fork_padding)
        # self.bin_conv = nn.Conv3d(in_channels=self.n_channels,
        #                           out_channels=self.n_bins,
        #                           kernel_size=(1, self.fork_kernel_size, self.fork_kernel_size),
        #                           padding=(0, self.fork_padding, self.fork_padding))
        # self.inv_conv = nn.Conv2d(in_channels=self.n_channels,
        #                           out_channels=1,
        #                           kernel_size=self.fork_kernel_size,
        #                           padding=self.fork_padding)

    # def forward(self, ipt):
    #     opt = ipt
    #     opt = self.conv_init(opt)
    #     for i in range(self.n_blocks):
    #         block = getattr(self, f"block{i}")
    #         opt = block(opt)
    #     opt_con = self.con_conv(opt).sigmoid()
    #     opt_bin = opt.unsqueeze(-3)
    #     opt_bin = opt_bin.expand(-1, -1, self.n_dists, -1, -1)
    #     opt_bin = self.bin_conv(opt_bin)
    #     opt_inv = self.inv_conv(opt).sigmoid()
    #     return opt_con, opt_bin, opt_inv

    def forward(self, ipt):
        opt = ipt
        opt = self.conv_init(opt)
        for i in range(self.n_blocks):
            block = getattr(self, f"block{i}")
            opt = block(opt)
        return opt
