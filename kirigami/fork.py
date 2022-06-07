import torch
import torch.nn as nn
from collections import namedtuple


class ForkHead(nn.Module):
    def __init__(self, n_channels, n_bins, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.bin_conv = nn.Conv2d(in_channels=n_channels,
                                  out_channels=n_bins,
                                  kernel_size=kernel_size,
                                  padding=padding)
        self.inv_conv = nn.Conv2d(in_channels=n_channels,
                                  out_channels=1,
                                  kernel_size=kernel_size,
                                  padding=padding)

    def forward(self, ipt):
        opt = {}
        opt["bin"] = self.bin_conv(ipt)
        opt["bin"] = (opt["bin"] + opt["bin"].transpose(-1, -2)) / 2
        opt["inv"] = self.inv_conv(ipt).sigmoid()
        opt["inv"] = (opt["inv"] + opt["inv"].transpose(-1, -2)) / 2
        return opt
        

class Fork(nn.Module):

    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]
    Opt = namedtuple("Opt", ["con"] + dist_types)

    def __init__(self, n_channels, n_bins, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.con_conv = nn.Conv2d(in_channels=n_channels,
                                  out_channels=1,
                                  kernel_size=kernel_size,
                                  padding=padding)
        for dist_type in self.dist_types:
            head = ForkHead(n_channels, n_bins, kernel_size)
            setattr(self, dist_type, head)

    def forward(self, ipt):
        opt = {}
        opt["dists"] = {}
        opt["con"] = self.con_conv(ipt).sigmoid()
        opt["con"] = (opt["con"] + opt["con"].transpose(-1, -2)) / 2
        for dist_type in self.dist_types:
            head = getattr(self, dist_type)
        opt["dists"][dist_type] = head(ipt)
        return opt

    # def forward(self, ipt):
    #     opt = self.con_conv(ipt).sigmoid()
    #     opt = (opt + opt.transpose(-1, -2)) / 2
    #     return opt
