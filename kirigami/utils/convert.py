import json
import re
from pathlib import Path
from typing import Tuple, OrderedDict, Optional
from collections import defaultdict, namedtuple, deque
from operator import itemgetter
from itertools import permutations
import torch
import torch.nn.functional as F
from kirigami._globals import *


__all__ = ["concatenate_batch",
           "concatenate_tensor",

           "sequence2sparse",
           "sequence2dense",
           "pairmap2sparse",
           "pairmap2dense",

           "seqlab2sparse",
           "seqlab2dense",

           "dense2pairmap",
           "dense2sequence",
           "dense2seqlab",
           "dense2bpseq",
           "seqlab2bpseq",

           "dotbracket2pairmap",
           # "label2sparse",
           # "label2dense",

           "st2seqlab",
           "st2sparse",
           "st2dense",

           "bpseq2seqlab",
           "bpseq2sparse",
           "bpseq2dense"]


def concatenate_batch(ipt: torch.tensor) -> torch.tensor:
    assert ipt.dim() == 3
    ipt_unsqz = ipt.unsqueeze(-1)
    rows = torch.cat(ipt_unsqz.shape[2] * [ipt_unsqz], 3)
    cols = rows.permute(0, 1, 3, 2)
    out = torch.cat((rows,cols), 1)
    return out


def concatenate_tensor(ipt: torch.tensor) -> torch.tensor:
    assert ipt.dim() == 2
    ipt_unsqz = ipt.unsqueeze(0)
    out = concatenate_batch(ipt_unsqz)
    return out.squeeze() 


def sequence2sparse(sequence: str,
                    dim: int = 3,
                    pad_length: int = 0,
                    dtype: torch.dtype = torch.uint8,
                    device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.fasta`-style sequence to sparse tensor"""
    sequence_copy = sequence.strip().upper()
    seq_length = len(sequence_copy)
    full_length = max(pad_length, seq_length)
    size = (4, full_length)
    while len(size) < dim:
        size = (1, *size)
    i = torch.zeros(2, seq_length, device=device)
    beg = (full_length - seq_length) // 2
    end = beg + seq_length
    i[1,:] = torch.arange(seq_length, device=device) + beg
    i[0,:] = torch.tensor([CHAR2IDX[char] for char in sequence_copy], device=device)
    v = torch.ones(i.shape[1], dtype=dtype, device=device)
    out = torch.sparse_coo_tensor(indices=i, values=v, size=size, dtype=dtype, device=device)
    return out


def sequence2dense(sequence: str,
                   dim: int = 3,
                   pad_length: int = 0,
                   dtype: torch.dtype = torch.uint8,
                   device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.fasta`-style sequence to tensor"""
    dense = sequence2sparse(sequence, dim, pad_length, dtype, device).to_dense()
    while dense.dim() < dim:
        dense.unsqueeze_(0)
    return dense


def pairmap2sparse(pair_map: PairMap,
                   dim: int = 3,
                   pad_length: int = 0,
                   dtype: torch.dtype = torch.uint8,
                   device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `PairMap` object to sparse tensor"""
    # drop unpaired nucleotides
    # sort every index-pair so redundancies are obvious (e.g., (0,99), (99,0))
    # i = list(map(sorted, i))
    # drop redundancies
    # i = list(set(map(tuple, i)))
    seq_length = len(pair_map)
    pad_length = max(pad_length, seq_length)
    size = (pad_length, pad_length)
    while len(size) < dim:
        size = (1, *size)
    beg = (pad_length - seq_length) // 2
    pairs = [pair for pair in pair_map.items() if pair[1] >= 0]
    if not pairs:
        # no pairs in molecule; need to pass `False` b/c at least one value required
        return torch.sparse_coo_tensor(indices=[[0],[0]], values=[False], size=size, dtype=dtype, device=device)
    i = torch.tensor(pairs, device=device).T + beg # add padding offset
    v = torch.ones(i.shape[1], dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices=i, values=v, size=size, dtype=dtype, device=device)


def pairmap2dense(pair_map: PairMap,
                  dim: int = 3,
                  pad_length: int = 0,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `PairMap` object to dense tensor"""
    dense = pairmap2sparse(pair_map, dim, pad_length, dtype, device).to_dense()
    while dense.dim() < dim:
        dense.unsqueeze_(0)
    return dense



def seqlab2sparse(seq_lab: SeqLab,
                  dim: int = 3,
                  pad_length: int = 0,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> Tuple[torch.tensor,torch.tensor]:
    """converts `SeqLab` object to sparse tensors"""
    seq_sparse = sequence2sparse(seq_lab.sequence, dim, pad_length, dtype, device)
    pair_map_sparse = pairmap2sparse(seq_lab.pair_map, dim, pad_length, dtype, device)
    return seq_sparse, pair_map_sparse 


def seqlab2dense(seq_lab: SeqLab,
                 dim: int = 3,
                 pad_length: int = 0,
                 dtype: torch.dtype = torch.uint8,
                 device: torch.device = torch.device("cpu")) -> Tuple[torch.tensor,torch.tensor]:
    """converts `SeqLab` object to dense tensors"""
    seq_dense = sequence2dense(seq_lab.sequence, dim, pad_length, dtype, device)
    pair_map_dense = pairmap2dense(seq_lab.pair_map, dim, pad_length, dtype, device)
    return seq_dense, pair_map_dense 


def dense2pairmap(ipt: torch.Tensor, seq_length: Optional[int] = None) -> PairMap:
    """converts binarized contact matrix to `PairMap` object"""
    mat = ipt.squeeze()
    assert mat.dim() == 2
    seq_length = seq_length or mat.shape[0]
    beg = (mat.shape[0] - seq_length) // 2
    end = beg + seq_length
    values, js = torch.max(mat[beg:end,beg:end], 0)
    js -= beg
    js[values <= 0.] = NO_CONTACT
    js_ints = map(int, js)
    pair_map = OrderedDict(enumerate(js_ints))
    return pair_map


def dense2sequence(ipt: torch.Tensor) -> str:
    """converts embedded `.fasta`-style sequence to string"""
    ipt_ = ipt.squeeze()
    total_length = ipt_.shape[1]
    seq_length = ipt_.sum().item()
    beg = (total_length - seq_length) // 2
    end = beg + seq_length
    _, js = torch.max(ipt_[:,beg:end], 0)
    return "".join([BASE_CHARS[j] for j in js])


def dense2seqlab(sequence: torch.Tensor, label: torch.Tensor) -> SeqLab:
    sequence_str = dense2sequence(sequence)
    pair_map = dense2pairmap(label)
    return SeqLab(len(sequence_str), sequence_str, pair_map)


def dense2bpseq(sequence: torch.Tensor, label: torch.Tensor) -> str:
    """converts sequence and label tensors to `.bpseq`-style string"""
    seq_lab = dense2seqlab(sequence, label)
    return seqlab2bpseq


def seqlab2bpseq(seq_lab: SeqLab) -> str:
    """converts `.fasta`-style sequence and `PairMap` to `.bpseq`-style string"""
    out_list = [f"{i+1} {char.upper()} {seq_lab.pair_map[i]+1}\n" for i, char in enumerate(seq_lab.sequence)]
    return "".join(out_list)



def dotbracket2pairmap(dot_bracket: str) -> PairMap:
    """converts `.db`-style string to `PairMap`"""
    lines = dot_bracket.splitlines()
    start_idx = 0
    while lines[start_idx].startswith("#"):
        start_idx += 1
    lines = lines[start_idx]
    paren = deque()
    square = deque()
    curly = deque()
    out = PairMap()
    for i, char in enumerate(lines):
        if char == ".":
            out[i] = NO_CONTACT
        elif char == "(":
            out[i] = NO_CONTACT
            paren.append(i)
        elif char == ")":
            j = paren.pop()
            out[i] = j
            out[j] = i
        elif char == "[":
            out[i] = NO_CONTACT
            square.append(i)
        elif char == "]":
            j = square.pop()
            out[i] = j
            out[j] = i
        elif char == "{":
            out[i] = NO_CONTACT
            curly.append(i)
        elif char == "}":
            j = curly.pop()
            out[i] = j
            out[j] = i
    return out
            

# def label2dense(label: str,
#                 pad_length: int = 0,
#                 dtype: torch.dtype = torch.uint8,
#                 device: torch.device = torch.device("cpu")) -> torch.tensor:
#     """Converts label file to contact matrix (`torch.Tensor`)"""
#     lines = label.splitlines()
#     matches = re.findall(r"[\d]+$", lines[0])
#     length = int(matches[0])
#     out = torch.zeros(length, length)
#     for line in lines:
#         if line.startswith("#") or line.startswith("i"):
#             continue
#         line_split = line.split()
#         idx1, idx2 = int(line_split[0]), int(line_split[-1])
#         out[idx1-1, idx2-1] = 1.
#     while out_dim > out.dim():
#         out.unsqueeze_(0)
#     return out



def st2seqlab(st: str) -> SeqLab:
    """converts `st`-style string to `SeqLab` object"""
    lines = st.splitlines()
    start_idx = 0
    while lines[start_idx].startswith("#"):
        start_idx += 1
    sequence = lines[start_idx]
    dot_bracket = lines[start_idx+1]
    pair_map = dotbracket2pairmap(dot_bracket)
    return SeqLab(len(pair_map), sequence, pair_map)


def st2sparse(st: str,
              pad_length: int = 0,
              dtype: torch.dtype = torch.uint8,
              device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.st`-style string to sparse tensor"""
    seq_lab = st2seqlab(st)
    seq_sparse = sequence2sparse(seq_lab.sequence, dim, pad_length, dtype, device)
    pair_map_sparse = pairmap2sparse(seq.pair_map, dim, pad_length, dtype, device)
    return seq_sparse, pair_map_sparse 
    

def st2dense(st: str,
             dim: int = 3,
             pad_length: int = 0,
             dtype: torch.dtype = torch.uint8,
             device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.st`-style string to dense tensor"""
    dense = seqlab2sparse(st2seqlab(st)).to_dense()
    while dense.dim() < dim:
        dense.unsqueeze_(0)
    return dense



def bpseq2seqlab(bpseq: str) -> SeqLab:
    """converts `.bpseq`-style string `SeqLab`"""
    lines = bpseq.splitlines()
    lines = list(filter(lambda line: not line.startswith("#"), lines))
    length = len(lines)
    pair_default = defaultdict(lambda: NO_CONTACT)
    sequence = ""
    for line in lines:
        i, base, j = line.split()
        i, j = int(i) - 1, int(j) - 1
        pair_default[i], pair_default[j] = j, i
        sequence += base.upper()
    pair_map = OrderedDict({i: pair_default[i] for i in range(length)})
    return SeqLab(len(sequence), sequence, pair_map)


def bpseq2sparse(bpseq: str,
                 dim: int = 3,
                 pad_length: int = 0,
                 dtype: torch.dtype = torch.uint8,
                 device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.bpseq`-style string to sparse tensor"""
    return seqlab2sparse(bpseq2seqlab(bpseq), dim, pad_length, dtype, device)


def bpseq2dense(bpseq: str,
                dim: int = 3,
                pad_length: int = 0,
                dtype: torch.dtype = torch.uint8,
                device: torch.device = torch.device("cpu")) -> torch.tensor:
    """Converts `.bpseq`-style string to dense tensor"""
    seq_dense, lab_dense = seqlab2dense(bpseq2seqlab(bpseq), dim, pad_length, dtype, device)
    while seq_dense.dim() < dim:
        seq_dense.unsqueeze_(0)
        lab_dense.unsqueeze_(0)
    return seq_dense, lab_dense
