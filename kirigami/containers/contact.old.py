from typing import *
from numbers import Real
from copy import copy
from dataclasses import dataclass, astuple
from collections import deque, defaultdict
from pathlib import Path
import torch


class Contact:

    @dataclass
    class Scores:
        tp: float
        tn: float
        fp: float
        fn: float
        f1: float
        mcc: float
        ground_pairs: int
        pred_pairs: int

    _StStr = NewType("_StStr", str)
    _BpseqStr = NewType("_BpseqStr", str)
    _Sequence = NewType("_Sequence", str)
    _DotBracket = NewType("_DotBracket", str)
    _ContactDict = Dict[int,int]
    _char_dict = {"A": 0, "U": 1, "C": 2, "G": 3}
    _idx_dict = {0: "A", 1: "U", 2: "C", 3: "G"}
    _NO_CONTACT = -1

    _pairs: _ContactDict
    _sequence: _Sequence

    def __init__(self,
                 pairs: _ContactDict,
                 sequence: _Sequence) -> None:
        self._pairs = pairs
        self._sequence = sequence
        # self.zuker_label = zuker_label

    @property
    def sequence(self):
        return self._sequence
    
    def __getitem__(self, ii: int) -> Optional[int]:
        return self._pairs.get(ii)

    def __len__(self) -> int:
        return len(self._sequence)

    def to_tensor(self,
                  dim: int = 3,
                  length: Optional[int] = None,
                  concatenate: bool = True,
                  sparse: bool = False,
                  dtype: torch.dtype = torch.float,
                  device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert dim >= 2
        return (self._embed_sequence(dim, length, concatenate, sparse, dtype, device),
                self._embed_pairs(dim, length, sparse, dtype, device))


    def _embed_sequence(self,
                        dim: int = 3,
                        length: Optional[int] = None,
                        concatenate: bool = False,
                        sparse: bool = False,
                        dtype: torch.dtype = torch.uint8,
                        device: torch.device = torch.device("cpu")) -> torch.Tensor:
        assert dim >= 2
        length = max(length, len(self)) if length else len(self)
        size = (4, length)
        size = (dim-len(size))*(1,) + size
        i = torch.zeros(dim, len(self), device=device, dtype=dtype)
        offset = (length - len(self)) // 2
        i[-2,:] = torch.tensor([self._char_dict[char] for char in self.sequence])
        i[-1,:] = offset + torch.arange(len(self))
        v = torch.ones(len(self), dtype=dtype, device=device)
        out = torch.sparse_coo_tensor(indices=i,
                                      values=v,
                                      size=size,
                                      dtype=dtype,
                                      device=device)
        
        if not sparse:
            out = out.to_dense()
            if concatenate:
                out = Contact.concatenate(out)
        return out

    def _embed_pairs(self, 
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

    @staticmethod
    def concatenate(ipt: torch.Tensor) -> torch.Tensor:
        """Performs outer concatenation on batch of sequence tensors. Implemented as
           static method to be reused outside `Contact` object in training 
           script"""
        out = ipt.unsqueeze(-1)
        out = torch.cat(out.shape[-2] * [out], dim=-1)
        # out = out.transpose(0, -3)
        out_t = out.transpose(-1, -2)
        out = torch.cat([out, out_t], dim=-3)
        return out 
         
    @classmethod
    def dotbracket2dict(dot_bracket: str) -> Dict[int,int]:
        paren = deque()
        square = deque()
        curly = deque()
        out = {}
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
        # out = defaultdict(lambda: None)
        out = {}
        for ii, row in enumerate(pairs_):
            jj = row.argmax().item()
            if row[jj] > 0:
                out[ii] = jj
        return out

    def get_scores(self, ground: "Contact") -> Scores: 
        """returns various evaluative scores of predicted secondary structure"""
        assert (length := len(self)) == len(ground_map)
        total = length * (length-1) / 2
        pred_set = set(self._pairs.items())
        ground_set = set(ground._pairs.items())
        pred_pairs, ground_pairs = len(pred_set), len(ground_set)
        tp = float(len(pred_set.intersection(ground_set)))
        fp = len(pred_set) - tp
        fn = len(ground_set) - tp
        tn = total - tp - fp - fn
        mcc = f1 = 0. 
        if len(pred_set) != 0 and len(ground_set) != 0:
            sn = tp / (tp+fn)
            pr = tp / (tp+fp)
            if tp > 0:
                f1 = 2*sn*pr / (pr+sn)
            if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
                mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
        return Scores(tp, tn, fp, fn, f1, mcc, ground_pairs, pred_pairs)
            
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
        # pairs = defaultdict(lambda: None)
        pairs = {}
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

    @staticmethod
    def tensor2sequence(ipt: torch.Tensor) -> str:
        ipt_ = ipt.squeeze()
        if ipt_.dim() == 3: # sequence is concatenated
            ipt_ = ipt_[:4,:,0]
        total_length = ipt_.shape[1]
        seq_length = int(ipt_.sum().item())
        beg = (total_length - seq_length) // 2
        end = beg + seq_length
        _, js = torch.max(ipt_[:,beg:end], 0)
        return "".join([Contact._idx_dict[j.item()] for j in js])

    @staticmethod
    def tensor2contactmap(ipt: torch.Tensor, seq_length: Optional[int] = None) -> _ContactDict:
        mat = ipt.squeeze()
        assert mat.dim() == 2
        seq_length = seq_length or mat.shape[0]
        beg = (mat.shape[0] - seq_length) // 2
        end = beg + seq_length
        values, js = torch.max(mat[beg:end,beg:end], 0)
        idxs = torch.arange(len(js))
        js = js[values > 0]
        idxs = idxs[values > 0]
        js_ints = map(int, js)
        idxs_ints = map(int, idxs)
        contact_dict = dict(zip(idxs_ints, js_ints))
        return contact_dict

    @classmethod
    def from_tensor(cls, tensors: Tuple[torch.Tensor, torch.Tensor]) -> "Contact":
        return Contact(sequence=Contact.tensor2sequence(tensors[0]),
                       pairs=Contact.tensor2contactmap(tensors[1]))        

# class Contact(defaultdict):
#     def __init__(self,
#                  pairs: Dict[int,int],
#                  length: Optional[int] = None) -> None:
#         super().__init__()
#         self.setdefault(lambda: 0)
#         self.update(pairs)
#         self.length = None
# 
#     def __len__(self) -> int:
#         return self.length or super().__len__()
# 
#     def embed(self,
#               dim: int = 3,
#               length: Optional[int] = None,
#               sparse: bool = False,
#               dtype: torch.dtype = torch.uint8,
#               device: torch.device = torch.device("cpu")) -> torch.Tensor:
#         assert dim >= 2
#         length = max(length, len(self)) if length else len(self)
#         size = (length, length)
#         size = (dim-len(size))*(1,) + size
#         offset = (length - len(self)) // 2
#         i = offset + torch.tensor(list(self._pairs.items()), device=device).T
#         i = torch.cat((torch.zeros(dim-i.dim(), i.shape[-1], device=device), i))
#         v = torch.ones(i.shape[-1], dtype=dtype, device=device)
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
