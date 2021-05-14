from typing import Optional, Tuple, List
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import *
import kirigami


__all__ = ["SpotNet"]


class SpotNet(torch.nn.Module):
    """implements SPOT-RNA architecture"""
    
    layers: torch.nn.Sequential
    
    def __init__(self,
                 n_res: int,
                 depth_res: int,
                 depth_lstm: int,
                 n_fc: int,
                 depth_fc: int,
                 p_res: float = 0.25,
                 p_fc: float = 0.5,
                 out_size: int = 512,
                 max_dilation_exp: int = 4,
                 act: str = "ELU",
                 res_norm: str = "BatchNorm2d",
                 fc_norm: str = "BatchNorm1d",
                 add_sigmoid: bool = False,
                 **kwargs) -> None:
        super().__init__()
        layer_dict = OrderedDict()
        # initial convolution
        layer_dict["conv"] = torch.nn.Conv2d(in_channels=8, out_channels=depth_res, kernel_size=3, padding=1)
        # Block A's 
        resnet_kwargs = kwargs if res_norm == "LayerNorm" else {"num_features": depth_res}
        dilations = [2**(i%max_dilation_exp) for i in range(2*n_res)]
        layer_dict["resnet"] = kirigami.nn.ResNet(n_blocks=n_res, p=p_res, dilations=dilations, n_channels=depth_res, **resnet_kwargs)
        # Bi-LSTM
        layer_dict["lstm"] = torch.nn.Sequential(kirigami.nn.ActNorm(act=act, norm=res_norm, **resnet_kwargs),
                                                 torch.nn.Conv2d(in_channels=depth_res, out_channels=1, kernel_size=1),
                                                 kirigami.nn.Squeeze(),
                                                 torch.nn.LSTM(input_size=out_size,
                                                               hidden_size=depth_lstm if n_fc > 0 else out_size,
                                                               batch_first=True,
                                                               bidirectional=True),
                                                 kirigami.nn.DropH0C0())
        # Block B's
        fc_kwargs = {"num_features": out_size}
        fc_list = [] 
        if n_fc > 0:
            # have to multiply `depth_lstm` by 2 because LSTM is bidirectional
            fc_list.extend([torch.nn.Linear(in_features=2*depth_lstm, out_features=depth_fc if n_fc > 1 else out_size),
                            kirigami.nn.ActNormDrop(act=act, norm=fc_norm, p=p_fc, **fc_kwargs)])
        if n_fc > 2:
            for _ in range(n_fc-2):
                fc_list.extend([torch.nn.Linear(depth_fc, depth_fc),
                                kirigami.nn.ActNormDrop(act=act, norm=fc_norm, p=p_fc, **fc_kwargs)])
        if n_fc > 1:
            fc_list.extend([torch.nn.Linear(depth_fc, out_size),
                            kirigami.nn.ActNormDrop(act=act, norm=fc_norm, p=p_fc, **fc_kwargs)])
        layer_dict["fc"] = torch.nn.Sequential(*fc_list)
        # output layer (optionally sigmoid)
        layer_dict["output"] = torch.nn.Sigmoid() if add_sigmoid else torch.nn.Identity()
        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        return self.layers(ipt)
