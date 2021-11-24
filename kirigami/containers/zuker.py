from typing import *
import torch

__all__ = ["Zuker"]

class Zuker(torch.Tensor):
    @classmethod
    def from_dict(cls,
                  pairs: Dict[Tuple[int,int], float],
                  dim: int = 2,
                  length: Optional[int] = None,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> "Zuker":
        length = max(length, len(pairs)) if length else len(pairs)
        out = torch.zeros((length, length), dtype=dtype, device=device)
        iis = list(map(lambda tup: tup[0] - 1, pairs.keys()))
        jjs = list(map(lambda tup: tup[1] - 1, pairs.keys()))
        out[iis, jjs] = torch.tensor(list(pairs.values()), device=device, dtype=dtype)
        return out.as_subclass(cls)

    def to_dict(self) -> Dict[Tuple[int,int], float]:
        iis_tens, jjs_tens = torch.argwhere(self)
        iis = list(map(int, iis_tens))
        jjs = list(map(int, jjs_tens))
        keys = list(zip(iis, jjs))
        vals = self[iis, jjs]
        items = zip(keys, vals)
        return dict(items)
