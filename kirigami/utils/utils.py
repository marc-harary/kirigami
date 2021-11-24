import functools
from typing import List, Callable
import torch
import torch.nn as nn

# __all__ = ["flatten_module", "compose_functions"]
__all__ = ["compose_functions"]

# def flatten_module(module: nn.Module) -> List[nn.Module]:
#     mods = []
#     for val in module._modules.values():
#         mods.extend(flatten_module(val))
#     return mods if mods else [module]

def compose_functions(functions: List[Callable]):
    return functools.reduce(lambda f, g: lambda x, y: f(*g(x,y)), functions, lambda x, y: (x, y))
