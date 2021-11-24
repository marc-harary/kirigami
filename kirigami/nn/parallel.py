from typing import *
import torch
from torch import nn

__all__ = ["Parallel"]

class Parallel(nn.Module):
    def __init__(*args, **kwargs):
        super().__init__()
        self.contact_mod = nn.Conv2d(*args)
        self.inv_mod = nn.Conv2d(*args)
        self.one_hot_mod = nn.Conv2d(*args)


    def forward(self, ipt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert ipt.dim() == 4
        return self.contact_mod(ipt[0]), self.inv_mod(ipt[0]), self.one_hot_mod(ipt[0])
