from typing import *
from copy import copy
from numbers import Real
from pathlib import Path
import torch
from kirigami.containers import Contact


__all__ = ["Zuker"]


class Zuker(Contact):
    def __init__(self,
                 contact_pairs: Contact._ContactDict,
                 sequence: Contact._Sequence,
                 zuker_pairs: Dict[Tuple[int,int], float]) -> None:
        super().__init__(contact_pairs, sequence)
        self._zuker_pairs = zuker_pairs

    @property
    def zuker_pairs(self) -> Dict[Tuple[int,int], float]:
        return self._zuker_pairs

    def _embed_zuker(self,
                     dim: int = 3,
                     length: Optional[int] = None,
                     sparse: bool = False,
                     dtype: torch.dtype = torch.float32,
                     device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 2
        length = max(length, len(self)) if length else len(self)
        out = torch.zeros((length, length), dtype=dtype, device=device)
        keys = self._zuker_pairs.keys()
        iis = list(map(lambda tup: tup[0] - 1, keys))
        jjs = list(map(lambda tup: tup[1] - 1, keys))
        out[iis, jjs] = torch.tensor(list(self._zuker_pairs.values()), device=device, dtype=dtype)
        return out 
        
    def to_tensor(self,
                  dim: int = 2,
                  length: Optional[int] = None,
                  concatenate: bool = True,
                  sparse: bool = False,
                  dtype: torch.dtype = torch.float,
                  device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, pairs = super().to_tensor(dim, length, concatenate, sparse, dtype, device)
        return self._embed_zuker(dim, length, sparse, dtype, device), sequence, pairs 

    @classmethod
    def from_zukers(cls, txt: str) -> "ZukerLabel":
        lines = copy(txt).splitlines()
        lines = list(filter(lambda line: not line.startswith("#"), lines))
        contact_pairs = {}
        length = len(lines)
        sequence = ""
        zuker_pairs = {}
        for line in lines:
            words = line.split()
            if words[1].isalpha(): # in regular bpseq-portion of file
                ii, base, jj = words
                ii, jj = int(ii) - 1, int(jj) - 1
                sequence += base.upper()
                if jj == -1:
                    continue
                contact_pairs[ii], contact_pairs[jj] = jj, ii
            else: # in Zuker-portion of file
                prob = float(words[2]) 
                if prob == 0:
                    continue
                ii, jj = list(map(int, words[:2]))
                zuker_pairs[ii, jj] = prob
        return cls(contact_pairs, sequence, zuker_pairs)

    @classmethod
    def from_zuker(cls, zuker_path: Path) -> "ZukerLabel":
        with open(zuker_path, "r") as f:
            txt = f.read()
        return cls.from_zukers(txt)


# class Zuker(defaultdict):
#     def __init__(self, data: Dict[int,float]) -> None:
#         super().__init__()
# 
#     def embed(self,
#               dim: int = 3,
#               length: Optional[int] = None,
#               sparse: bool = False,
#               dtype: torch.dtype = torch.float32,
#               device: torch.device = torch.device("cpu")) -> torch.Tensor:
#         assert dim >= 2
#         length = max(length, len(self)) if length else len(self)
#         size = (length, length)
#         size = (dim-len(size))*(1,) + size
#         offset = (length - len(self)) // 2
#         i = offset + torch.tensor(list(self._pairs.keys()), device=device).T
#         i = torch.cat((torch.zeros(dim-i.dim(), i.shape[-1], device=device), i))
#         v = torch.tensor(self._pairs.values(), dtype=dtype, device=device)
#         if len(v) > 0:
#             out = torch.sparse_coo_tensor(indices=i, 
#                                           values=v,
#                                           size=size,
#                                           dtype=dtype,
#                                           device=device)
#         else:
#             out = torch.sparse_coo_tensor(size=size,
#                                           dtype=dtype,
#                                           device=device) 
#         return out if sparse else out.to_dense()
