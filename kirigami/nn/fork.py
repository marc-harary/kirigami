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
            # conv = ResNet(in_channels=32, n_blocks=1, n_channels=32)
            conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=n_bins if multiclass else 1,
                                   kernel_size=kernel_size,
                                   padding=1)
            # conv = ResNet(in_channels=in_channels, n_blocks=2, p=0., norm="InstanceNorm2d", n_channels=32)
            setattr(self, f"dist{i:01}", conv)
        self.dist_conv = torch.nn.Conv2d(in_channels=32,
                                         out_channels=n_bins if multiclass else 1,
                                         kernel_size=kernel_size,
                                         padding=1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, ipt: torch.Tensor):
        lab = self.label_conv(ipt)
        opt = [self.sigmoid(lab)]
        for i in range(self.out_dists):
            conv = getattr(self, f"dist{i:01}")
            opt.append(conv(ipt))
            # opt.append(self.soft_max(conv(ipt)))
            # opt.append(self.soft_max(self.dist_conv(conv(ipt))))
        return tuple(opt)


# class Fork(AtomicModule):
#     def __init__(self,
#                  in_channels: int,
#                  kernel_size: int,
#                  out_dists: int = 10,
#                  multiclass: bool = False,
#                  n_bins: int = 1):
#         super().__init__()
#         self.label_conv = torch.nn.Conv2d(in_channels=in_channels,
#                                           out_channels=1,
#                                           kernel_size=kernel_size,
#                                           padding=1)
#         self.out_dists = out_dists
#         self.dist_conv1 = torch.nn.Conv2d(in_channels=in_channels,
#                                          out_channels=out_dists,
#                                          kernel_size=kernel_size,
#                                          padding=1)
#         self.dist_conv2 = torch.nn.Conv3d(in_channels=1,
#                                          out_channels=n_bins,
#                                          kernel_size=kernel_size,
#                                          padding=1)
#         self.sigmoid = nn.Sigmoid()
#         self.soft_max = nn.Softmax(dim=1)
# 
#     
#     def forward(self, ipt: torch.Tensor):
#         lab = self.label_conv(ipt)
#         # opt = [self.sigmoid(lab)]
#         dists = self.dist_conv1(ipt)
#         dists = dists.unsqueeze(1)
#         dists = self.dist_conv2(dists)
#         dists = self.soft_max(dists)
#         dists = torch.tensor_split(dists, self.out_dists, dim=2)
#         dists = [dist.squeeze(2) for dist in dists]
#         return self.sigmoid(lab), *dists
