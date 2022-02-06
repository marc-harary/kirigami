from typing import Optional, Tuple, List
import torch
from torch.nn import *
from kirigami.nn.regularize import ActNormDrop


__all__ = ["ResNetBlock", "ResNet"]


class ResNetBlock(torch.nn.Module):
    """implements ResNet unit"""
    resnet: bool
    conv1: Module
    act_norm_drop1: Module
    conv2: Module
    act_norm_drop2: Module
    def __init__(self,
                 p: float,
                 dilations: Tuple[int,int],
                 kernel_sizes: Tuple[int],
                 act: str,
                 norm: str,
                 n_channels: int,
                 resnet: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.resnet = resnet
        self.act_norm_drop1 = ActNormDrop(p=p,
                                          act=act,
                                          norm=norm,
                                          **kwargs)
        # self.act_norm_drop1 = eval(act)()
        self.conv1 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[0],
                                     dilation=dilations[0],
                                     padding=self.get_padding(dilations[0], kernel_sizes[0]))
        self.act_norm_drop2 = ActNormDrop(p=p,
                                          act=act,
                                          norm=norm,
                                          **kwargs)
        # self.act_norm_drop2 = eval(act)()
        self.conv2 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels,
                                     kernel_size=kernel_sizes[1],
                                     dilation=dilations[1],
                                     padding=self.get_padding(dilations[1], kernel_sizes[1]))

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act_norm_drop1(out)
        out = self.conv1(out)
        out = self.act_norm_drop2(out)
        out = self.conv2(out)
        if self.resnet:
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
                 in_channels: int = 8,
                 n_channels: int = 8,
                 p: float = 0.5,
                 dilations: Optional[List[int]] = None,
                 kernel_sizes: Tuple[int,int] = (3,5),
                 act: str = "ELU",
                 include_sigmoid: bool = True,
                 norm: str = "BatchNorm2d",
                 resnet: bool = True,
                 **kwargs) -> None:
        super().__init__()
        if norm in ["BatchNorm2d", "InstanceNorm2d"]:
            kwargs["num_features"] = n_channels
        self.conv_init = torch.nn.Conv2d(in_channels=in_channels,
                                         out_channels=n_channels,
                                         kernel_size=kernel_sizes[0],
                                         padding=1)
        block_list = []
        dilations = dilations or 2*n_blocks*[1]
        assert len(dilations) == 2*n_blocks, "Must pass in two dilations per block!"
        for i in range(n_blocks):
            block =  ResNetBlock(p=p,
                                 dilations=dilations[2*i:2*(i+1)],
                                 kernel_sizes=kernel_sizes,
                                 act=act,
                                 norm=norm,
                                 n_channels=n_channels,
                                 resnet=resnet,
                                 **kwargs) 
            block_list.append(block)
        self.blocks = Sequential(*block_list) 
        self.conv_final = torch.nn.Conv2d(in_channels=n_channels,
                                          out_channels=1,
                                          kernel_size=kernel_sizes[0],
                                          padding=1)
        self.final_act = Sigmoid() if include_sigmoid else Identity()
        
    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        opt = self.conv_init(ipt)
        opt = self.blocks(opt)
        opt = self.conv_final(opt)
        opt = self.final_act(opt)
        return opt
