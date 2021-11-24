from typing import *
from torch import nn

__all__ = ["Triple", "sequentialize", "AtomicModule"]

T = TypeVar("T")
Triple = Tuple[T,T,T]

class AtomicModule(nn.Module):
    pass
    
def _sequentialize(module: nn.Module) -> List[nn.Module]:
    """recursively breaks `nn.Module` into sub-modules unless `module` is 
       of type `AtomicModule`"""
    modules = []
    if not isinstance(module, AtomicModule):
        for val in module._modules.values():
            modules.extend(_sequentialize(val))
    return modules if modules else [module]

def sequentialize(in_modules: Sequence[nn.Module]) -> nn.Sequential:
    out_modules = []
    for module in in_modules:
       out_modules.extend(_sequentialize(module))
    model = nn.Sequential(*out_modules)
    return model
