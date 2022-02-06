import torch
from torch import nn
from kirigami.nn.utils import *
import torch.nn.functional as F


__all__ = ["Fork"]


class Fork(AtomicModule):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 out_dist_channels: int = 10):
        super().__init__()
        self.label_conv = torch.nn.Conv2d(in_channels=in_channels,
                                          out_channels=1,
                                          kernel_size=kernel_size,
                                          padding=1)
        self.dist_conv = torch.nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_dist_channels,
                                         kernel_size=kernel_size,
                                         padding=1)
        self.sigmoid = nn.Sigmoid()
        # self.act_norm_drop = ActNormDrop(act="ELU",
        #                                  norm="",
        #                                  p=
        
    
    def forward(self, ipt: torch.Tensor):
        lab = self.label_conv(ipt)
        lab = self.sigmoid(lab)
        dist = self.dist_conv(ipt)
        dist = self.sigmoid(dist)
        # dist = F.relu(dist)
        return lab, dist
