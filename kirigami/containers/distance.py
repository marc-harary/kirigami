from typing import *
from numbers import *
from dataclasses import dataclass
import torch
from abc import ABC


class Distance(ABC):

    @dataclass
    class DistancePair:
        PP: float = 0.
        O5O5: float = 0. 
        C5C5: float = 0.
        C4C4: float = 0.
        C3C3: float = 0.
        C2C2: float = 0.
        C1C1: float = 0.
        O4O4: float = 0.
        O3O3: float = 0.
        NN: float = 0.

    @classmethod
    def from_dict(cls,
                  pairs: Dict[int,int],
                  dim: int = 2,
                  length: Optional[int] = None,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> "Distance":
        pass

    def to_dict(self) -> Dict[Tuple[int,int], DistancePair]:
        pass 



class BinDistance(Distance):
    @classmethod
    def from_dict(cls,
                  pairs: Dict[int,int],
                  bins: Sequence[Real],
                  dim: int = 4,
                  length: Optional[int] = None,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> "BinDistance":
        assert dim >= 4
        if not isinstance(bins, torch.Tensor):
            bins_ = torch.tensor(bins)
        else:
            bins_ = bins

        idxs = torch.tensor(list(pairs.keys())).T
        native_length = idxs.max().item() + 1
        length = max(length, native_length) if length else native_length
        # shape = (len(bins)+1, 10, length, length)
        # shape = (dim-len(shape))*(1,) + shape
        shape = (10, length, length)
        out = torch.zeros(shape, dtype=dtype, device=device)
        offset = (length-native_length) // 2
        dist_tups = list(map(astuple, pairs.values()))
        dist_tens = torch.tensor(dist_tups, dtype=dtype, device=device)
        out[:, idxs[:,0], idxs[:,1]] = dist_tens.T
        out = out.repeat(len(bins_), 1, 1, 1)
        bins_ = bins_.reshape(1, 1, 1, len(bins_)).T
        print(torch.where(out > bins_))
        vals, idxs = torch.max(dist_tens <= bins_, 0) # gets leftmost `True` if any
        idxs[vals == 0] = -1
        print(idxs)
        
        # for i in range(len(self)):
        #     for j in range(len(self)):
        #         pair = self[i,j]
        #         for k, dist in enumerate(astuple(pair)):
        #             val, idx = torch.max(dist <= bins_, 0) # gets leftmost `True` if any
        #             idx = idx if val else -1 # max is `False` so dist > last bin
        #             out[...,idx,k,i+offset,j+offset] = 1.
        # return out

    @classmethod
    def to_dict(self) -> Dict[Tuple[int,int], Distance.DistancePair]:
        pass



class InvDistance(Distance):
    @classmethod
    def from_dict(cls,
                  pairs: Dict[int,int],
                  A: Real = 1.0, 
                  eps: float = 1e-4,
                  dim: int = 3,
                  length: Optional[int] = None,
                  dtype: torch.dtype = torch.float,
                  device: torch.device = torch.device("cpu")) -> "InvDistance":
        idxs = torch.tensor(list(pairs.keys()))
        native_length = idxs.max().item() + 1
        length = max(length, native_length) if length else native_length
        shape = (10, length, length)
        shape = (dim-len(shape))*(1,) + shape
        out = torch.zeros(shape, dtype=dtype, device=device)
        dist_tups = list(map(astuple, pairs.values()))
        dist_tens = torch.tensor(dist_tups, dtype=dtype, device=device)
        out[:, idxs[:,0], idxs[:,1]] = dist_tens.T
        out += eps
        out = A / out
        return out

    @classmethod
    def to_dict(self) -> Dict[Tuple[int,int], Distance.DistancePair]:
        pass
