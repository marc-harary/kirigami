import torch
from torch import nn
from kirigami.nn.utils import *


__all__ = ["Fork"]


class Fork(AtomicModule):

    contact_module: nn.Module
    inv_module: nn.Module
    one_hot_module: nn.Module

    def __init__(self,
                 contact_module: nn.Module,
                 inv_module: nn.Module,
                 one_hot_module: nn.Module) -> None:
        super().__init__()
        self.contact_module = contact_module
        self.inv_module = inv_module
        self.one_hot_module = one_hot_module

    def forward(self, ipt: torch.Tensor) -> Triple[torch.Tensor]:
        return (self.contact_module(ipt),
                self.inv_module(ipt),
                self.one_hot_module(ipt))
