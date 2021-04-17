import json
import re
from pathlib import Path
from typing import Tuple, OrderedDict
from collections import defaultdict, namedtuple, deque
from operator import itemgetter
from itertools import permutations
import torch
from kirigami._globals import *


__all__ = ["st2pairmap",
           "dotbracket2pairmap",
           "bpseq2pairmap",
           "tensor2pairmap",
           "pairmap2tensor",
           "sequence2tensor",
           "label2tensor",
           "bpseq2tensor",
           "st2tensor",
           "tensor2sequence",
           "tensor2bpseq",
           "pairmap2bpseq"]


def pairmap2tensor(pairs: PairMap, out_dim: int = 3) -> torch.Tensor:
    """Converts `PairMap` to contact matrix (`torch.Tensor`)"""
    length = len(pairs)
    out = torch.zeros(length, length)
    for i, j in pairs.items():
        if j == NO_CONTACT:
            continue
        out[i,j] = 1.
    while out_dim > out.dim():
        out.unsqueeze_(0)
    return out


def sequence2tensor(sequence: str) -> torch.Tensor:
    """Converts `FASTA` sequence to `torch.Tensor`"""
    sequence_copy = sequence.strip().upper()
    length = len(sequence_copy)
    one_hot = torch.stack([CHAR2TENSOR[char] for char in sequence_copy])
    rows = one_hot.T
    rows = rows.unsqueeze(2)
    rows = rows.expand(-1, -1, length)
    cols = rows.permute(0, 2, 1)
    out = torch.vstack((rows,cols))
    return out


def dotbracket2pairmap(dot_bracket: str) -> PairMap:
    """Converts dot bracket-style files to `PairMap`"""
    lines = dot_bracket.splitlines()
    start_idx = 0
    while lines[start_idx].startswith("#"):
        start_idx += 1
    lines = lines[start_idx]
    paren = deque()
    square = deque()
    curly = deque()
    out = {}
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
            

def label2tensor(label: str, out_dim: int = 4) -> torch.Tensor:
    """Converts label file to contact matrix (`torch.Tensor`)"""
    lines = label.splitlines()
    matches = re.findall(r"[\d]+$", lines[0])
    length = int(matches[0])
    out = torch.zeros(length, length)
    for line in lines:
        if line.startswith("#") or line.startswith("i"):
            continue
        line_split = line.split()
        idx1, idx2 = int(line_split[0]), int(line_split[-1])
        out[idx1-1, idx2-1] = 1.
    while out_dim > out.dim():
        out.unsqueeze_(0)
    return out


def bpseq2pairmap(bpseq: str) -> Tuple[str, PairMap]:
    """Converts `.bpseq` file to string and `PairMap`"""
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
    pair_map = {i: pair_default[i] for i in range(length)}
    return sequence, pair_map


def st2pairmap(st: str) -> Tuple[str, PairMap]:
    """Converts `.st` file to string and `PairMap`"""
    lines = st.splitlines()
    start_idx = 0
    while lines[start_idx].startswith("#"):
        start_idx += 1
    sequence = lines[start_idx]
    dot_bracket = lines[start_idx+1]
    return sequence, dotbracket2pairmap(dot_bracket)


def st2tensor(st: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts `.st` file to string and `PairMap`"""
    sequence, pair_map = st2pairmap(st)
    return sequence2tensor(sequence), pairmap2tensor(pair_map)
    

def bpseq2tensor(bpseq: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts `.bpseq` file to `torch.Tensor`"s"""
    sequence, pair_map = bpseq2pairmap(bpseq)
    return sequence2tensor(sequence), pairmap2tensor(pair_map)


def pairmap2bpseq(sequence: str, pair_map: PairMap) -> str:
    """Converts `FASTA`-style sequence and `PairMap` to `.bpseq`-style string"""
    assert len(sequence) == len(pair_map)
    out_list = [f"{i+1} {char.upper()} {pair_map[i]+1}\n" for i, char in enumerate(sequence)]
    return "".join(out_list)


def tensor2pairmap(ipt: torch.Tensor) -> PairMap:
    """Converts binarized contact matrix to `PairMap`"""
    mat = ipt.squeeze()
    assert mat.dim() == 2
    values, js = torch.max(mat, 1)
    js[values == 0.] = NO_CONTACT
    js_ints = map(int, js)
    pair_map = OrderedDict(enumerate(js_ints))
    return pair_map


def tensor2sequence(ipt: torch.Tensor) -> str:
    """Converts embedded `FASTA` sequence to string"""
    chars_embed = ipt.squeeze()
    chars_embed = chars_embed[:N_BASES, :, 0].T
    chars = []
    for row in chars_embed:
        _, idx = torch.max(row, 0)
        chars.append(BASE_CHARS[idx])
    return "".join(chars)


def tensor2bpseq(sequence: torch.Tensor, label: torch.Tensor) -> str:
    """Converts sequence and label tensors to `bpseq`-like string"""
    sequence_str = tensor2sequence(sequence)
    label_pair_map = tensor2pairmap(label)
    return pairmap2bpseq(sequence_str, label_pair_map)
