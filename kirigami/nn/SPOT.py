import torch
from torch import nn

__all__ = ['ActDropNorm', 'BlockA', 'BlockB']


class ActDropNorm(nn.Module):
    '''Performs activation, dropout, and batch normalization for resnet blocks'''
    def __init__(self, p: float, activation='ReLU', num_channels=8):
        super(ActDropNorm, self).__init__()
        activation_class = getattr(nn, activation)
        self.act = activation_class()
        self.drop = nn.Dropout(p=p)
        self.norm = nn.BatchNorm2d(num_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        out = self.act(out)
        out = self.drop(out)
        out = self.norm(out)
        return out


class BlockA(nn.Module):
    '''Implements BlockA resnet from SPOT-RNA network'''
    def __init__(self,
                 p: float,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        out = self.act_drop_norm(out)
        out = self.conv1(out)
        out = self.act_drop_norm(out)
        out = self.conv2(out)
        if self.resnet:
            out += input
        return out


class BlockB(nn.Module):
    '''Implements BlockB FC layers from SPOT-RNA network'''
    def __init__(self,
                 p: float,
                 activation='ReLU',
                 in_features=8,
                 out_features=8):
        super(BlockB, self).__init__()
        self.lin = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_drop_norm = ActDropNorm(num_channels=out_features,
                                         p=p,
                                         activation=activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        out = self.lin(out)
        out = self.act_drop_norm(out)
        return out
