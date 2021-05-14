from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.nn import *


__all__ = ["ActNormDrop", "ActNorm"]


class ActNorm(nn.Module):
    """performs activation, dropout, and normalization for resnet blocks"""
    
    act: Module
    norm: Module 

    def __init__(self,
                 act: str,
                 norm: str,
                 **kwargs) -> None:
        super().__init__()
        self.act = eval(act)()
        self.norm = eval(norm)(**kwargs)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act(out)
        if self.training:
            out = self.norm(out)
        return out


class ActNormDrop(ActNorm):
    """performs activation, dropout, and normalization for resnet blocks"""
    
    act: Module
    norm: Module 
    drop: Module

    def __init__(self,
                 act: str,
                 norm: str,
                 p: float,
                 **kwargs) -> None:
        super().__init__(act, norm, **kwargs)
        self.act = eval(act)()
        self.norm = eval(norm)(**kwargs)
        self.drop = nn.Dropout2d(p=p)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = ipt
        out = self.act(out)
        if self.training:
            out = self.norm(out)
            out = self.drop(out)
        return out
