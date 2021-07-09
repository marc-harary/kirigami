from typing import *
from numbers import Real
from copy import copy
from math import ceil
from dataclasses import dataclass, astuple
from collections import defaultdict
from pathlib import Path
import torch

__all__ = ["Distance"]



class Distance:

    @dataclass
    class Pair:
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

    _PdbStr = NewType("_PdbStr", str)
    _StStr = NewType("_StStr", str)
    _Sequence = NewType("_Sequence", str)
    _N_ATOMS = len(astuple(Pair()))

    _pairs: OrderedDict[Tuple[int,int], Pair]
    _sequence: _Sequence

    def __init__(self, pairs: OrderedDict[Tuple[int,int],Pair], sequence: _Sequence) -> None:
        self._pairs = pairs
        self._sequence = sequence

    @property
    def sequence(self):
        return self._sequence
    
    def __getitem__(self, idx: Tuple[int,int]) -> Pair:
        return self._pairs[idx]

    def __len__(self) -> int:
        return len(self._sequence)

    def _to_float(self,
                  dim: int = 3,
                  length: Optional[int] = None,
                  dtype: torch.dtype = torch.float,
                  device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 3
        dtype = torch.float32
        length = max(length, len(self))
        shape = (10, length, length)
        shape = (dim-len(shape))*(1,) + shape
        out_tensor = torch.ones(shape, device=device, dtype=dtype)
        for i in range(len(self)):
            for j in range(len(self)):
                out_tensor[...,:,i,j] = torch.tensor(astuple(self[i,j]))
        return out_tensor

    def to_inv(self,
               A: Real = 1.0, 
               eps: float = 1e-4,
               dim: int = 3,
               length: Optional[int] = 0,
               dtype: torch.dtype = torch.float,
               device: torch.device = torch.device("cpu")) -> torch.Tensor:
        out = self._to_float(dim, length, dtype, device)
        out[out < 0.] = 0. 
        out += eps
        out = A / out
        return out
    
   #  def to_one_hot(self,
   #                 max_dist: Real = 22.0, 
   #                 bin_width: Real = 0.5,
   #                 dim: int = 4,
   #                 length: Optional[int] = None,
   #                 dtype: torch.dtype = torch.uint8,
   #                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
   #      assert dim >= 4
   #      length = max(length, len(self)) if length else len(self)
   #      offset = (length-len(self)) // 2
   #      num_bins = int(max_dist//bin_width + 1)
   #      shape = (num_bins, len(self.Pair.__annotations__), length, length)
   #      shape = (dim-len(shape))*(1,) + shape
   #      out = torch.zeros(shape, dtype=dtype, device=device)
   #      for i in range(len(self)):
   #          for j in range(len(self)):
   #              pair = self[i,j]
   #              for k, dist in enumerate(astuple(pair)):
   #                  dist = min(dist, max_dist) # clip between 0 and max_dist
   #                  idx = ceil(dist/bin_width) - 1
   #                  idx = max(idx, 0)
   #                  out[...,idx,k,i+offset,j+offset] = 1.
   #      return out

    # def to_one_hot(self,
    #                bins: Sequence[Real],
    #                dim: int = 4,
    #                length: Optional[int] = None,
    #                dtype: torch.dtype = torch.uint8,
    #                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    #     assert dim >= 4
    #     if not isinstance(bins, torch.Tensor):
    #         bins_ = torch.tensor(bins)
    #     else:
    #         bins_ = bins
    #     length = max(length, len(self)) if length else len(self)
    #     offset = (length-len(self)) // 2
    #     shape = (len(bins)+1, len(self.Pair.__annotations__), length, length)
    #     shape = (dim-len(shape))*(1,) + shape
    #     out = torch.zeros(shape, dtype=dtype, device=device)
    #     for i in range(len(self)):
    #         for j in range(len(self)):
    #             pair = self[i,j]
    #             for k, dist in enumerate(astuple(pair)):
    #                 val, idx = torch.max(dist <= bins_, 0) # gets leftmost `True` if any
    #                 idx = idx if val else -1 # max is `False` so dist > last bin
    #                 out[...,idx,k,i+offset,j+offset] = 1.
    #     return out

    def to_one_hot(self,
                   bins: Sequence[Real],
                   dim: int = 4,
                   length: Optional[int] = None,
                   sparse: bool = False,
                   dtype: torch.dtype = torch.uint8,
                   device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 4
        if not isinstance(bins, torch.Tensor):
            bins_ = torch.tensor(bins)
        else:
            bins_ = bins
        length = max(length, len(self)) if length else len(self)
        offset = (length-len(self)) // 2
        shape = (len(bins)+1, self._N_ATOMS, length, length)
        dim_diff = dim - len(shape)
        shape = dim_diff*(1,) + shape
        tot = self._N_ATOMS * len(self)**2
        idxs = torch.zeros((tot, len(shape)), dtype=torch.int, device=device)
        vals = torch.ones(tot)
        idx_row = 0
        for i in range(len(self)):
            for j in range(len(self)):
                pair = self[i,j]
                for k, dist in enumerate(astuple(pair)):
                    val, idx = torch.max(dist <= bins_, 0) # gets leftmost `True` if any
                    idxs[idx_row,-4] = idx if val else len(bins)
                    idxs[idx_row,-3] = k
                    idxs[idx_row,-2] = i + offset 
                    idxs[idx_row,-1] = j + offset 
                    idx_row += 1
        out = torch.sparse_coo_tensor(indices=idxs.T, values=vals, size=shape, device=device, dtype=dtype)
        return out if sparse else out.to_dense()

    def to_tensor(self, 
                  bins: Sequence[Real],
                  A: Real = 1.0, 
                  eps: float = 1e-4,
                  # max_dist: Real = 22.0, 
                  # bin_width: Real = 0.5,
                  dim: int = 4,
                  length: Optional[int] = None,
                  sparse: bool = False,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.to_one_hot(bins=bins,
                                dim=dim,
                                length=length,
                                sparse=sparse,
                                dtype=dtype,
                                device=device),
               self.to_inv(A=A,
                           eps=eps,
                           dim=dim,
                           length=length,
                           dtype=dtype,
                           device=device))
            
    @classmethod
    def from_txts(cls, pdb: _PdbStr) -> "Distance":
        lines = copy(pdb).splitlines()
        del lines[0] # drop header
        length = int(0.5 * (1 + (1 + 2*4*len(lines))**.5)) # quadratic formula
        sequence = ""
        pairs = defaultdict(lambda: cls.Pair())
        for i, line in enumerate(lines):
            words = line.split()
            # need to shift modulo b/c file doesn't list redundant pairs
            # (i.e., no j-i after i-j has been liste)
            if length and not i % length:
                sequence += words[0].upper()
                length -= 1
            j = int(words[2]) - 1
            k = int(words[5]) - 1
            dist_list = list(map(float, words[-10:]))
            dist = Distance.Pair(*dist_list)
            pairs[(j,k)] = pairs[(k,j)] = dist
        return cls(pairs, sequence)

    @classmethod
    def from_txt(cls, pdb_path: Path) -> "Distance":
        with open(pdb_path, "r") as f:
            txt = f.read()
        return cls.from_txts(txt) 

    @classmethod
    def from_file(cls, txt_path: Path) -> "Distance":
        if txt_path.endswith("txt"):
            return cls.from_txt(txt_path)
        else:
            raise ValueError("Invalid file type")
