from typing import *
from numbers import Real
from copy import copy
from dataclasses import dataclass, astuple
from collections import deque, defaultdict
from pathlib import Path
import torch


class Contact:

    _StStr = NewType("_StStr", str)
    _BpseqStr = NewType("_BpseqStr", str)
    _Sequence = NewType("_Sequence", str)
    _DotBracket = NewType("_DotBracket", str)
    _ContactDict = DefaultDict[Tuple[int,int], Optional[int]]
    _char_dict = {"A": 0, "U": 1, "C": 2, "G": 3}
    _idx_dict = {0: "A", 1: "U", 2: "C", 3: "G"}

    _pairs: _ContactDict
    _sequence: _Sequence

    def __init__(self,
                 pairs: _ContactDict,
                 sequence: _Sequence) -> None:
        self._pairs = pairs
        self._sequence = sequence

    @property
    def sequence(self):
        return self._sequence
    
    def __getitem__(self, idx: Tuple[int,int]) -> bool:
        return (ii := self._pairs[idx[0]]) and ii == idx[1] 

    def __len__(self) -> int:
        return len(self._sequence)

    def to_tensor(self,
                  dim: int = 3,
                  length: Optional[int] = None,
                  concatenate: bool = True,
                  sparse: bool = False,
                  dtype: torch.dtype = torch.float,
                  device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        assert dim >= 3
        return (self._embed_sequence(dim, length, concatenate, sparse, dtype, device),
                self._embed_pairs(dim, length, concatenate, sparse, dtype, device))

    def _embed_sequence(self,
                        dim: int = 2,
                        length: Optional[int] = None,
                        concatenate: bool = True,
                        sparse: bool = False,
                        dtype: torch.dtype = torch.uint8,
                        device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 2
        length = max(length, len(self)) if length else len(self)
        size = (4, length)
        size = (dim-len(size))*(1,) + size
        i = torch.zeros(2, length, device=device, dtype=dtype)
        offset = (length - len(self)) // 2
        i[0,offset:length+offset] = torch.tensor([self._char_dict[char] for char in self.sequence])
        i[1,:] = offset + torch.arange(len(self))
        v = torch.ones(i.shape[1], dtype=dtype, device=device)
        out = torch.sparse_coo_tensor(indices=i,
                                      values=v,
                                      size=size,
                                      dtype=dtype,
                                      device=device)
        if not sparse:
            out = out.to_dense()
            if concatenate:
                out = out.unsqueeze(-1).repeat(*(1,)*dim, length)
                out_t = out.transpose(-1, -2)
                out = torch.cat((out, out_t), dim=-3)
        return out

    def _embed_pairs(self, 
                     dim: int = 2,
                     length: Optional[int] = None,
                     sparse: bool = False,
                     dtype: torch.dtype = torch.uint8,
                     device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 2
        length = max(length, len(self)) if length else len(self)
        size = (length, length)
        size = (dim-len(size))*(1,) + size
        offset = (length - len(self)) // 2
        pairs = []
        for i in range(len(self)):
            for j in range(len(self)):
               if self[i,j]:
                   pairs.append((i,j))
        i = offset + torch.tensor(pairs, device=device).T
        v = torch.ones(i.shape[1], dtype=dtype, device=device)
        out = torch.sparse_coo_tensor(indices=i, 
                                      values=v,
                                      size=size,
                                      dtype=dtype,
                                      device=device)
        return out if sparse else out.to_dense()

    @classmethod
    def dotbracket2dict(dot_bracket: str) -> _ContactDict:
        paren = deque()
        square = deque()
        curly = deque()
        out = _ContactDict(lambda i: None)
        for i, char in enumerate(lines):
            if char == "(":
                paren.append(i)
            elif char == ")":
                j = paren.pop()
                out[i] = j
                out[j] = i
            elif char == "[":
                square.append(i)
            elif char == "]":
                j = square.pop()
                out[i] = j
                out[j] = i
            elif char == "{":
                curly.append(i)
            elif char == "}":
                j = curly.pop()
                out[i] = j
                out[j] = i
        return out

    @staticmethod
    def tensor2sequence(seq: torch.Tensor) -> _Sequence:
        seq_ = copy(seq).squeeze()
        if seq_.dim() == 3:
            row = seq[4:,0,:]  
        elif seq_.dim() == 2:
            row = seq
        else:
            raise ValueError("Ill-formed tensor")
        idxs = torch.argmax(row, axis=0).tolist()
        chars = [Contact._idx_dict[idx] for idx in idxs]
        return "".join(chars)

    @staticmethod
    def tensor2contactdict(pairs: torch.Tensor) -> _ContactDict:
        pairs_ = copy(pairs).squeeze()
        if pairs_.dim() != 2:
            raise ValueError("Ill-formed tensor")
        out = defaultdict(lambda: None)
        for ii, row in enumerate(pairs_):
            jj = row.argmax().item()
            if row[jj] > 0:
                out[ii] = jj
        return out
            
    @classmethod
    def from_sts(cls, st: _StStr) -> "Contact":
        start_idx = 0
        while lines[start_idx].startswith("#"):
            start_idx += 1
        sequence = lines[start_idx]
        dot_bracket = lines[start_idx+1]
        contact_map = cls.dotbracket2dict(dot_bracket)
        return cls(contact_map, sequence)

    @classmethod
    def from_st(cls, st_path: Path) -> "Contact":
        with open(st_path, "r") as f:
            txt = f.read()
        return cls.from_sts(txt)

    @classmethod
    def from_bpseqs(cls, bpseq: _BpseqStr) -> "Contact":
        lines = copy(bpseq).splitlines()
        lines = list(filter(lambda line: not line.startswith("#"), lines))
        pairs = defaultdict(lambda: None)
        length = len(lines)
        sequence = ""
        for line in lines:
            i, base, j = line.split()
            i, j = int(i) - 1, int(j) - 1
            sequence += base.upper()
            if j == -1:
                continue
            pairs[i], pairs[j] = j, i
        return cls(pairs, sequence)

    @classmethod
    def from_bpseq(cls, bpseq_path: Path) -> "Contact":
        with open(bpseq_path, "r") as f:
            txt = f.read()
        return cls.from_bpseqs(txt)

    @classmethod
    def from_file(cls, txt_path: Path) -> "Contact":
        if txt_path.endswith("st"):
            return cls.from_st(txt_path)
        elif txt_path.endswith("bpseq"):
            return cls.from_bpseq(txt_path)
        else:
            raise ValueError("File format not recognized")
