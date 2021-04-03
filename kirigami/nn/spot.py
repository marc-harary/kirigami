'''implements sub-modules from Singh et al. (2020)'''


import torch
from torch import nn
from kirigami.nn.ResNet import ActDropNorm


__all__ = ['BlockA', 'BlockB']


class BlockA(nn.Module):
    '''Implements BlockA resnet from SPOT-RNA network'''
    def __init__(self,
                 p: float,
                 activation='ReLU',
                 n_channels=8,
                 kernel_size1=3,
                 kernel_size2=5,
                 resnet=True):
        super().__init__()
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

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act_drop_norm(out)
        out = self.conv1(out)
        out = self.act_drop_norm(out)
        out = self.conv2(out)
        if self.resnet:
            out += ipt
        return out


class BlockB(nn.Module):
    '''Implements BlockB FC layers from SPOT-RNA network'''
    def __init__(self,
                 p: float,
                 activation='ReLU',
                 in_features=8,
                 out_features=8):
        super().__init__()
        self.lin = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_drop_norm = ActDropNorm(num_channels=out_features,
                                         p=p,
                                         activation=activation)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.lin(out)
        out = self.act_drop_norm(out)
        return out
