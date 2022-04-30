from typing import *
import torch
from torch import nn
from kirigami.nn.utils import *
import torch.nn.functional as F
from kirigami.nn.resnet import ResNet


__all__ = ["Fork"]


class Fork(AtomicModule):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 out_dists: int = 10,
                 multiclass: bool = False,
                 n_bins: int = 1):
        super().__init__()
        self.label_conv = torch.nn.Conv2d(in_channels=in_channels,
                                          out_channels=1,
                                          kernel_size=kernel_size,
                                          padding=1)
        self.out_dists = out_dists
        for i in range(out_dists):
            conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=n_bins if multiclass else 1,
                                   kernel_size=kernel_size,
                                   padding=1)
            setattr(self, f"bin_dist{i:01}", conv)
        for i in range(out_dists):
            conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=1,
                                   kernel_size=kernel_size,
                                   padding=1)
            setattr(self, f"inv_dist{i:01}", conv)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ipt: torch.Tensor):
        lab = self.sigmoid(self.label_conv(ipt))
        bin_dists = []
        for i in range(self.out_dists):
            conv = getattr(self, f"bin_dist{i:01}")
            bin_dists.append(conv(ipt))
        inv_dists = []
        for i in range(self.out_dists):
            conv = getattr(self, f"inv_dist{i:01}")
            inv_dists.append(conv(ipt).sigmoid())
        return lab, tuple(bin_dists), tuple(inv_dists)
