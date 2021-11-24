from typing import *
from numbers import Real
from collections import defaultdict
from pathlib import Path
import torch


__all__ = ["Contact"]


class Contact(defaultdict):
    def __init__(self,
                 pairs: Dict[int,int],
                 length: Optional[int] = None) -> None:
        super().__init__()
        self.setdefault(lambda: 0)
        self.update(pairs)
        self.length = None

    def __len__(self) -> int:
        return self.length or super().__len__()

    def embed(self,
              dim: int = 3,
              length: Optional[int] = None,
              sparse: bool = False,
              dtype: torch.dtype = torch.uint8,
              device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 2
        length = max(length, len(self)) if length else len(self)
        size = (length, length)
        size = (dim-len(size))*(1,) + size
        offset = (length - len(self)) // 2
        i = offset + torch.tensor(list(self._pairs.items()), device=device).T
        i = torch.cat((torch.zeros(dim-i.dim(), i.shape[-1], device=device), i))
        v = torch.ones(i.shape[-1], dtype=dtype, device=device)
        if len(v) > 0:
            out = torch.sparse_coo_tensor(indices=i, 
                                          values=v,
                                          size=size,
                                          dtype=dtype,
                                          device=device)
        else:
            out = torch.sparse_coo_tensor(size=size,
                                          dtype=dtype,
                                          device=device) 
        return out if sparse else out.to_dense()
