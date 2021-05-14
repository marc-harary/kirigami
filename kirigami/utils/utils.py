import functools
from typing import List
import torch
import torch.nn as nn

__all__ = ["flatten_module"]

def flatten_module(module: nn.Module) -> List[nn.Module]:
    mods = []
    for val in module._modules.values():
        mods.extend(flatten_module(val))
    return mods if mods else [module]
