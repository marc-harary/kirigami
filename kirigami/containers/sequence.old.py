from typing import *
from numbers import Real
from copy import copy
from dataclasses import dataclass, astuple
from collections import deque, defaultdict
from enum import *
from pathlib import Path
import torch


class Seq(torch.Tensor):

    _char_dict = {"A": 0, "U": 1, "C": 2, "G": 3}

    def __init__(self, bases: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bases = bases

    @property
    def bases(self) -> str:
        return self._bases

    def concatenate(self) -> torch.Tensor:
        """Performs outer concatenation on batch of sequence tensors. Implemented as
           static method to be reused outside `Contact` object in training 
           script"""
        out = ipt.unsqueeze(-1)
        out = torch.cat(out.shape[-2] * [out], dim=-1)
        out_t = out.transpose(-1, -2)
        out = torch.cat([out, out_t], dim=-3)
        return out 

    @classmethod
    def from_str(cls,
                 bases: str,
                 dim: int = 2,
                 length: Optional[int] = None,
                 concatenate: bool = False,
                 dtype: torch.dtype = torch.uint8,
                 device: torch.device = torch.device("cpu")) -> "Seq":
        assert dim >= 2
        length = max(length, len(bases)) if length else len(bases)
        size = (4, length)
        size = (dim-len(size))*(1,) + size
        out = torch.zeros(size, device=device, dtype=dtype)
        offset = (length - len(bases)) // 2
        idxs = [Seq._char_dict[char] for char in bases]
        out[..., idxs, list(range(offset,offset+length))] = 1.
        out = out.as_subclass(cls)
        if concatenate:
            out = self.concatenate()
        out._bases = bases
        return out
