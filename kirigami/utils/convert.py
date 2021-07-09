import json
import re
from pathlib import Path
from typing import Tuple, OrderedDict, Optional
from collections import defaultdict, namedtuple, deque
from math import ceil
import torch
import torch.nn.functional as F
from kirigami._globals import *
from kirigami._classes import *


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
           "bpseq2dense",

           "pdb2float",
           "pdb2inv",
           "pdb2bin",

           "pdb2distmap",
           "distmap2float",
           "distmap2inv",
           "distmap2bin"]


def concatenate_batch(ipt: torch.tensor, dim: int = 4) -> torch.tensor:
    assert ipt.dim() == 3
    ipt_unsqz = ipt.unsqueeze(-1)
    rows = torch.cat(ipt_unsqz.shape[2] * [ipt_unsqz], 3)
    cols = rows.permute(0, 1, 3, 2)
    out = torch.cat((rows,cols), 1)
    while out.dim() < dim:
        out.unsqueeze_(0)
    return out


def concatenate_tensor(ipt: torch.tensor, dim: int = 3) -> torch.tensor:
    ipt_sqz = ipt.squeeze()
    assert ipt_sqz.dim() == 2
    ipt_unsqz = ipt_sqz.unsqueeze(0)
    out = concatenate_batch(ipt_unsqz)
    while out.dim() < dim:
        out.unsqueeze_(0)
    return out


# def sequence2sparse(sequence: str,
#                     dim: int = 2,
#                     pad_length: int = 0,
#                     dtype: torch.dtype = torch.uint8,
#                     device: torch.device = torch.device("cpu")) -> torch.tensor:
#     """converts `.fasta`-style sequence to sparse tensor"""
#     sequence_copy = sequence.strip().upper()
#     seq_length = len(sequence_copy)
#     full_length = max(pad_length, seq_length)
#     size = (4, full_length)
#     while len(size) < dim:
#         size = (1, *size)
#     i = torch.zeros(2, full_length, device=device)
#     beg = (full_length - seq_length) // 2
#     end = beg + seq_length
#     i[0,beg:end] = torch.tensor([CHAR2IDX[char] for char in sequence_copy], device=device)
#     i[1,beg:end] = torch.arange(seq_length, device=device) + beg
#     v = torch.ones(i.shape[1], dtype=dtype, device=device)
#     while i.shape[0] < dim:
#         i = torch.cat([torch.zeros(1,full_length), i])
#     out = torch.sparse_coo_tensor(indices=i, values=v, size=size, dtype=dtype, device=device)
#     return out


def sequence2sparse(sequence: str,
                    dim: int = 2,
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
    i[0,:] = torch.tensor([CHAR2IDX[char] for char in sequence_copy], device=device)
    i[1,:] = torch.arange(seq_length, device=device) + beg
    v = torch.ones(i.shape[1], dtype=dtype, device=device)
    while i.shape[0] < dim:
        i = torch.cat([torch.zeros(1,seq_length), i])
    out = torch.sparse_coo_tensor(indices=i, values=v, size=size, dtype=dtype, device=device)
    return out


def sequence2dense(sequence: str,
                   dim: int = 2,
                   pad_length: int = 0,
                   dtype: torch.dtype = torch.uint8,
                   device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.fasta`-style sequence to tensor"""
    return sequence2sparse(sequence, dim, pad_length, dtype, device).to_dense()


def pairmap2sparse(contact_map: ContactMap,
                   dim: int = 2,
                   pad_length: int = 0,
                   dtype: torch.dtype = torch.uint8,
                   device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `ContactMap` object to sparse tensor"""
    # drop unpaired nucleotides
    # sort every index-pair so redundancies are obvious (e.g., (0,99), (99,0))
    # i = list(map(sorted, i))
    # drop redundancies
    # i = list(set(map(tuple, i)))
    seq_length = len(contact_map)
    pad_length = max(pad_length, seq_length)
    size = (pad_length, pad_length)
    while len(size) < dim:
        size = (1, *size)
    beg = (pad_length - seq_length) // 2
    pairs = [pair for pair in contact_map.items() if pair[1] >= 0]
    if not pairs:
        # no pairs in molecule; need to pass `False` b/c at least one value required
        return torch.sparse_coo_tensor(indices=torch.zeros(dim,1), values=[False], size=size, dtype=dtype, device=device)
    i = torch.tensor(pairs, device=device).T + beg # add padding offset
    v = torch.ones(i.shape[1], dtype=dtype, device=device)
    while i.shape[0] < dim:
        i = torch.cat([torch.zeros(1, len(pairs)), i])
    return torch.sparse_coo_tensor(indices=i, values=v, size=size, dtype=dtype, device=device)


def pairmap2dense(contact_map: ContactMap,
                  dim: int = 2,
                  pad_length: int = 0,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `ContactMap` object to dense tensor"""
    return pairmap2sparse(contact_map, dim, pad_length, dtype, device).to_dense()


def seqlab2sparse(seq_lab: SeqLab,
                  dim: int = 2,
                  pad_length: int = 0,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> Tuple[torch.tensor,torch.tensor]:
    """converts `SeqLab` object to sparse tensors"""
    seq_sparse = sequence2sparse(seq_lab.sequence, dim, pad_length, dtype, device)
    contact_map_sparse = pairmap2sparse(seq_lab.contact_map, dim, pad_length, dtype, device)
    return seq_sparse, contact_map_sparse 


def seqlab2dense(seq_lab: SeqLab,
                 dim: int = 2,
                 pad_length: int = 0,
                 dtype: torch.dtype = torch.uint8,
                 device: torch.device = torch.device("cpu")) -> Tuple[torch.tensor,torch.tensor]:
    """converts `SeqLab` object to dense tensors"""
    seq_dense = sequence2dense(seq_lab.sequence, dim, pad_length, dtype, device)
    contact_map_dense = pairmap2dense(seq_lab.contact_map, dim, pad_length, dtype, device)
    return seq_dense, contact_map_dense 


def dense2pairmap(ipt: torch.Tensor, seq_length: Optional[int] = None) -> ContactMap:
    """converts binarized contact matrix to `ContactMap` object"""
    mat = ipt.squeeze()
    assert mat.dim() == 2
    seq_length = seq_length or mat.shape[0]
    beg = (mat.shape[0] - seq_length) // 2
    end = beg + seq_length
    values, js = torch.max(mat[beg:end,beg:end], 0)
    # js -= beg
    js[values <= 0.] = NO_CONTACT
    js_ints = map(int, js)
    contact_map = OrderedDict(enumerate(js_ints))
    return contact_map


def dense2sequence(ipt: torch.Tensor) -> str:
    """converts embedded `.fasta`-style sequence to string"""
    ipt_ = ipt.squeeze()
    total_length = ipt_.shape[1]
    seq_length = int(ipt_.sum().item())
    beg = (total_length - seq_length) // 2
    end = beg + seq_length
    _, js = torch.max(ipt_[:,beg:end], 0)
    return "".join([BASE_CHARS[j] for j in js])


def dense2seqlab(sequence: torch.Tensor, label: torch.Tensor) -> SeqLab:
    sequence_str = dense2sequence(sequence)
    contact_map = dense2pairmap(label)
    return SeqLab(len(sequence_str), sequence_str, contact_map)


def dense2bpseq(sequence: torch.Tensor, label: torch.Tensor) -> str:
    """converts sequence and label tensors to `.bpseq`-style string"""
    seq_lab = dense2seqlab(sequence, label)
    return seqlab2bpseq


def seqlab2bpseq(seq_lab: SeqLab) -> str:
    """converts `.fasta`-style sequence and `ContactMap` to `.bpseq`-style string"""
    out_list = [f"{i+1} {char.upper()} {seq_lab.contact_map[i]+1}\n" for i, char in enumerate(seq_lab.sequence)]
    return "".join(out_list)


def dotbracket2pairmap(dot_bracket: str) -> ContactMap:
    """converts `.db`-style string to `ContactMap`"""
    lines = dot_bracket.splitlines()
    start_idx = 0
    while lines[start_idx].startswith("#"):
        start_idx += 1
    lines = lines[start_idx]
    paren = deque()
    square = deque()
    curly = deque()
    out = ContactMap()
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
    contact_map = dotbracket2pairmap(dot_bracket)
    return SeqLab(len(contact_map), sequence, contact_map)


def st2sparse(st: str,
              dim: int = 2,
              device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.st`-style string to sparse tensor"""
    seq_lab = st2seqlab(st)
    seq_sparse = sequence2sparse(seq_lab.sequence, dim, pad_length, dtype, device)
    contact_map_sparse = pairmap2sparse(seq.contact_map, dim, pad_length, dtype, device)
    return seq_sparse, contact_map_sparse 
    

def st2dense(st: str,
             dim: int = 2,
             pad_length: int = 0,
             dtype: torch.dtype = torch.uint8,
             device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.st`-style string to dense tensor"""
    return seqlab2sparse(st2seqlab(st), dim, pad_length, dtype, device).to_dense()


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
    contact_map = OrderedDict({i: pair_default[i] for i in range(length)})
    return SeqLab(len(sequence), sequence, contact_map)


def bpseq2sparse(bpseq: str,
                 dim: int = 2,
                 pad_length: int = 0,
                 dtype: torch.dtype = torch.uint8,
                 device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.bpseq`-style string to sparse tensor"""
    return seqlab2sparse(bpseq2seqlab(bpseq), dim, pad_length, dtype, device)


def bpseq2dense(bpseq: str,
                dim: int = 2,
                pad_length: int = 0,
                dtype: torch.dtype = torch.uint8,
                device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `.bpseq`-style string to dense tensor"""
    return seqlab2dense(bpseq2seqlab(bpseq), dim, pad_length, dtype, device)


def pdb2distmap(pdb: str) -> DistMap:
    lines = pdb.splitlines()
    del lines[0] # drop header
    c = -2*len(lines) + 2
    dis = 1 - 4*c
    sqrt_val = dis**.5
    L = int((1+sqrt_val) / 2)
    out_unsort = {}
    for line in lines:
        words = line.split()
        i = int(words[2]) - 1
        j = int(words[5]) - 1
        dist_list = list(map(float, words[-10:]))
        dist = Dist(*dist_list)
        out_unsort[(i,j)] = out_unsort[(j,i)] = dist
    for i in range(L+1):
        out_unsort[(i,i)] = Dist(*(10*[0]))
    out = OrderedDict({key: out_unsort[key] for key in sorted(out_unsort)})
    return out

    
def distmap2float(dist_map: DistMap,
                  dim: int = 3,
                  pad_length: int = 0,
                  dtype: torch.dtype = torch.float,
                  device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `DistMap` to float-based tensor"""
    fields = dist_map[(0,0)].__annotations__.keys()
    L = int(len(dist_map)**.5)
    out = torch.zeros((10, L, L))
    for (i, j), dist in dist_map.items():
        out[:,i,j] = torch.tensor([getattr(dist, field) for field in fields])
    while out.dim() < dim:
        out.unsqueeze_(0)
    return out


def distmap2inv(dist_map: DistMap,
                A: int = 1.0,
                eps: float = 1e-4,
                dim: int = 3,
                pad_length: int = 0,
                dtype: torch.dtype = torch.float,
                device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `DistMap` object to inverted float-based tensor"""
    out = distmap2float(dist_map, dim, pad_length, dtype, device)
    out += eps
    out = A / out
    return out


# def distmap2bin(dist_map: DistMap,
#                 max_dist: float = 22.0, 
#                 bin_width: float = 0.5,
#                 dim: int = 4,
#                 pad_length: int = 0,
#                 dtype: torch.dtype = torch.uint8,
#                 device: torch.device = torch.device("cpu")) -> torch.tensor:
#     """converts `DistMap` object to one-hot encoded tensor"""
#     actual_length = int(len(dist_map)**.5)
#     pad_length = max(pad_length, actual_length)
#     diff_length = pad_length - actual_length
#     num_bins = int(max_dist/bin_width) + 1
#     out = torch.zeros((num_bins, N_ATOM_PAIRS, pad_length, pad_length), dtype=dtype, device=device)
#     for (j, k), dist in dist_map.items():
#         j, k = j+diff_length, k+diff_length
#         for i, atom_pair in enumerate(ATOM_PAIRS):
#             pair_dist = getattr(dist, atom_pair)
#             pair_dist = min(max(0, pair_dist), max_dist) # clip between 0 and max_dist
#             idx = int(pair_dist / bin_width)
#             out[idx, i, j, k] = True
#     while out.dim() < dim:
#         out.unsqueeze_(0)
#     return out


def distmap2bin(dist_map: DistMap,
                max_dist: float = 22.0, 
                bin_width: float = 0.5,
                dim: int = 4,
                pad_length: int = 0,
                dtype: torch.dtype = torch.uint8,
                device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `DistMap` object to one-hot encoded tensor"""
    actual_length = int(len(dist_map)**.5)
    pad_length = max(pad_length, actual_length)
    diff_length = pad_length - actual_length
    beg = diff_length // 2
    num_bins = int(max_dist//bin_width + 1)
    full_size = size = (num_bins, N_ATOM_PAIRS, pad_length, pad_length)
    if (diff := len(full_size) - dim) > 0:
        full_size = full_size + diff*(1,)
    out = torch.zeros(size, dtype=dtype, device=device)
    # idx = torch.zeros((N_ATOM_PAIRS,pad_length,pad_length), device=device)
    # idx[1,:] = idx[2:] = torch.arange(pad_length, device=device) + beg
    # v = torch.ones(idx.shape[1], dtype=dtype, device=device)
    for (j, k), dist in dist_map.items():
        j, k = j+diff_length, k+diff_length
        for i, atom_pair in enumerate(ATOM_PAIRS):
            pair_dist = getattr(dist, atom_pair)
            pair_dist = min(pair_dist, max_dist) # clip between 0 and max_dist
            # idx[i, j, k] = int(pair_dist // bin_width)
            idx = ceil(pair_dist/bin_width) - 1
            idx = max(idx, 0)
            out[..., idx, i, j, k] = 1.
    return out


def pdb2float(pdb: str,
              dim: int = 3,
              pad_length: int = 0,
              dtype: torch.dtype = torch.float,
              device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `PDB` file-basd text to float-based tensor"""
    return pdb2float(pdb2distmap(pdb), dim, pad_length, dtype, device)


def pdb2inv(pdb: str,
            A: int = 1.0,
            eps: float = 1e-4,
            dim: int = 3,
            pad_length: int = 0,
            dtype: torch.dtype = torch.float,
            device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `PDB` file-basd text to inverted float-based tensor"""
    return pdb2inv(pdb2distmap(pdb), A, eps, dim, pad_length, dtype, device)


def pdb2bin(pdb: str,
            max_dist: float = 22.0, 
            bin_width: float = 0.5,
            dim: int = 4,
            pad_length: int = 0,
            dtype: torch.dtype = torch.uint8,
            device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `PDB` filed-based text to one-hot encoded tensor"""
    return pdb2bin(pdb2distmap(pdb), max_dist, bin_width, dim, pad_length, dtype, device)


def pdb2invbin(pdb: str,
               A: int = 1.0,
               eps: float = 1e-4,
               inv_dim: int = 3,
               bin_dim: int = 4,
               max_dist: float = 22.0, 
               bin_width: float = 0.5,
               pad_length: int = 0,
               dtype: torch.dtype = torch.uint8,
               device: torch.device = torch.device("cpu")) -> torch.tensor:
    """converts `PDB` filed-based text to both float-based and one-hot encoded tensors"""
    return (pdb2inv(pdb, A, eps, inv_dim, pad_length, dtype, device),
        pdb2bin(pdb, max_dist, bin_width, bin_dim, pad_length, dtype, device))
