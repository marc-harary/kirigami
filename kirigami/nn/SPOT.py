import torch
from torch import nn


class ActDropNorm(nn.Module):
    def __init__(self, p, activation='ReLU', num_channels=8):
        super(ActDropNorm, self).__init__()
        activation_class = getattr(nn, activation)
        self.act = activation_class()
        self.drop = nn.Dropout(p=p)
        self.norm = nn.BatchNorm2d(num_channels)

    def forward(self, input):
        ret = input
        ret = self.act(ret)
        ret = self.drop(ret)
        ret = self.norm(ret)
        return ret


class BlockA(nn.Module):
    def __init__(self,
                 p,
                 activation='ReLU',
                 n_channels=8,
                 kernel_size1=3,
                 kernel_size2=5,
                 resnet=True):
        super(BlockA, self).__init__()
        self.resnet = resnet
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size1,
                               padding=kernel_size1//2)
        self.conv2 = nn.Conv2d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=kernel_size2,
                               padding=kernel_size2//2)
        self.act_drop_norm = ActDropNorm(num_channels=n_channels,
                                         p=p,
                                         activation=activation)

    def forward(self, input):
        ret = input
        ret = self.act_drop_norm(ret)
        ret = self.conv1(ret)
        ret = self.act_drop_norm(ret)
        ret = self.conv2(ret)
        if self.resnet:
            ret += input
        return ret


class BlockB(nn.Module):
    def __init__(self,
                 p,
                 activation='ReLU',
                 in_features=8,
                 out_features=8):
        super(BlockB, self).__init__()
        self.lin = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_drop_norm = ActDropNorm(num_channels=out_features,
                                         p=p,
                                         activation=activation)

    def forward(self, input):
        ret = input
        ret = self.lin(ret)
        ret = self.act_drop_norm(ret)
        return ret
