from typing import Optional, Tuple, List
import torch
import torch.nn as nn



__all__ = ["ResNetBlock", "ResNet"]



class ActNormDrop(nn.Module):
    """performs activation, dropout, and normalization for resnet blocks"""
    def __init__(self, p: float, num_features: int) -> None:
        super().__init__()
        self.act = nn.ELU()
        self.norm = nn.InstanceNorm2d(num_features=num_features)
        self.drop = nn.Dropout2d(p=p)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act(out)
        # if self.training:
        out = self.norm(out)
        out = self.drop(out)
        return out



class ResNetBlock(nn.Module):
    """implements ResNet unit"""
    def __init__(self,
                 p: float,
                 dilations: Tuple[int,int],
                 kernel_sizes: Tuple[int],
                 n_channels: int) -> None:
        super().__init__()
        self.act_norm_drop1 = ActNormDrop(p=p, num_features=n_channels)
        self.conv1 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[0],
                                     dilation=dilations[0],
                                     padding=self.get_padding(dilations[0], kernel_sizes[0]))
        self.act_norm_drop2 = ActNormDrop(p=p, num_features=n_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[1],
                                     dilation=dilations[1],
                                     padding=self.get_padding(dilations[1], kernel_sizes[1]))

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.conv1(out)
        out = self.act_norm_drop1(out)
        out = self.conv2(out)
        out = self.act_norm_drop2(out)
        out += ipt
        return out

    @staticmethod
    def get_padding(dilation: int, kernel_size: int) -> int:
        """returns padding needed for 'same'-like feature in TensorFlow"""
        return round((dilation * (kernel_size - 1)) / 2)



class ResNet(torch.nn.Module):
    """Implements ResNet"""
    def __init__(self,
                 n_blocks: int,
                 in_channels: int = 9,
                 n_channels: int = 32,
                 p: float = 0.5,
                 kernel_sizes: tuple = (3,5)):
        super().__init__()
        self.conv_init = torch.nn.Conv2d(in_channels=in_channels,
                                         out_channels=n_channels,
                                         kernel_size=kernel_sizes[0],
                                         padding=1)
        self.n_blocks = n_blocks
        dilations = 2 * n_blocks * [1]
        for i in range(n_blocks):
            block = ResNetBlock(p=p,
                                dilations=dilations[2*i:2*(i+1)],
                                kernel_sizes=kernel_sizes,
                                n_channels=n_channels)
            setattr(self, f"block{i}", block)
        

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        opt = self.conv_init(ipt)
        for i in range(self.n_blocks):
            block = getattr(self, f"block{i}")
            opt = block(opt)
        return opt
