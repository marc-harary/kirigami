import torch
from kirigami.nn import SPOT
from torch import nn

class BlockA(nn.Module):
    def __init__(self,
                 p,
                 n_channels=8,
                 resnet=True,
                 activation='ReLU',
                 kernel_size1=3,
                 kernel_size2=5):
        super(BlockA, self).__init__()
        activation_class = getattr(nn, activation)
        self.resnet = resnet
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size1,
                               padding=kernel_size1//2)
        self.conv2 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size2,
                               padding=kernel_size2//2)
        self.act = activation_class()
        self.drop = nn.Dropout(p)
        self.norm = nn.BatchNorm2d(n_channels)

    def forward(self, input):
        ret = input
        ret = self.act(ret)
        ret = self.norm(ret)
        ret = self.drop(ret)
        ret = self.conv1(ret)
        ret = self.act(ret)
        ret = self.norm(ret)
        ret = self.drop(ret)
        ret = self.conv2(ret)
        return ret + input if self.resnet else ret
