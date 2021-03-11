import json
import re
from pathlib import Path
from typing import Tuple
from collections import defaultdict, namedtuple, deque
from operator import itemgetter
from itertools import permutations

import torch
import munch

from kirigami._globals import *


__all__ = ["path2munch",
           "pairmap2tensor",
           "sequence2tensor",
           "label2tensor",
           "bpseq2tensor",
           "st2tensor",
           "dotbracket2pairmap",
           "tensor2pairmap",
           "tensor2sequence",
           "tensor2bpseq",
           "bpseq2pairmap",
           "st2pairmap",
           "pairmap2bpseq",
           "binarize",
           "get_scores"]


def path2munch(path: Path) -> munch.Munch:
    """Reads .json file saved at PATH and returns `Munch` object"""
    with open(path, "r") as f:
        txt = f.read()
    conf_json = json.loads(txt)
    conf =  munch.munchify(conf_json)
    return conf


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
    one_hot = torch.stack([BASE_DICT[char] for char in sequence_copy])
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
    brack = deque()
    out = []
    for i, char in enumerate(lines):
        if char == ".":
            out.append(NO_CONTACT)
        elif char == "(":
            out.append(NO_CONTACT)
            paren.append(i)
        elif char == ")":
            j = paren.pop()
            out.append(j)
            out[j] = i
        elif char == "[":
            out.append(NO_CONTACT)
            brack.append(i)
        elif char == "]":
            j = brack.pop()
            out.append(j)
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
    sequence, pair_map = str2pairmap(st)
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


def binarize(ipt: torch.Tensor,
             thres: float = .5,
             diagonal: float = 0.,
             min_dist: int = 4,
             canonicalize: bool = True) -> torch.Tensor:
    """Binarizes contact matrix from deep network"""
    # mat = ipt.squeeze()
    # length = mat.shape[0]
    # assert mat.dim() == 2 and length == mat.shape[1], "Input tensor must be square"


    # def filter_closure(base_pair, prob):
    #     i, j = base_pair
    #     not_too_close = (abs(i - j) >= 4)
    #     not_too_low = prob >= thres
    #     base_i, base_j = INV_BASE_DICT[i], INV_BASE_DICT[j]
    #     canonical = ((base_i == "G" and base_j == "C") or
    #                  (base_j == "G" and base_i == "C") or 
    #                  (base_i == "A" and base_j == "U") or 
    #                  (base_j == "A" and base_i == "U"))
    #    canonical = canonical if canonicalize else True
    #    return not_too_close and not_too_low and canonical

    # base_pairs = permutations(range(length), 2) # get all base combinations
    # probs = [mat[base_pair] for base_pair in base_pairs]
    # idx_and_probs = zip(base_pairs, probs)
    # sort(idx_and_probs, key=itemgetter(1))

    # idx_dict = OrderedDict()
    # for base_pair, prob in zip(base_pairs, probs):
    #     if filter_closure(base_pair, prob): 
    #         idx_dict[base_pair] = prob
    #     
    # while idx_dict:
    #     idx_pair = idx_dict.pop()
    #     i, j = val[0]
    #     out[i,j], out[j,i] = 1., 1.
    #     vals = list(filter(lambda val: not set((i,j)).intersection(set(val[0])), vals))

    # out.fill_diagonal_(diagonal)
    # return out


def tensor2pairmap(ipt: torch.Tensor) -> PairMap:
    """Converts binarized contact matrix to `PairMap`"""
    mat = ipt.squeeze()
    assert mat.dim() == 2
    values, js = torch.max(mat, 1)
    js[values == 0.] = NO_CONTACT
    js_ints = map(int, js)
    pair_map = dict(enumerate(js_ints))
    return pair_map


def tensor2sequence(ipt: torch.Tensor) -> str:
    """Converts embedded `FASTA` sequence to string"""
    chars_embed = ipt.squeeze()
    chars_embed = chars_embed[:N_BASES, :, 0].T
    chars = []
    for row in chars_embed:
        _, idx = torch.max(row, 0)
        chars.append(BASES[idx])
    return "".join(chars)


def tensor2bpseq(sequence: torch.Tensor, label: torch.Tensor) -> str:
    """Converts sequence and label tensors to `bpseq`-like string"""
    sequence_str = tensor2sequence(sequence)
    label_pair_map = tensor2pairmap(label)
    return pairmap2bpseq(sequence_str, label_pair_map)


def get_scores(pred_map: PairMap, ground_map: PairMap) -> Scores: 
    """Returns various evaluative scores of predicted secondary structure"""
    length = len(pred_map)
    assert length == len(ground_map)
    total = length * (length-1) / 2
    pred_set = {pair for pair in pred_map.items() if pair[1] >= pair[0]}
    ground_set = {pair for pair in ground_map.items() if pair[1] >= pair[0]}
    pred_pairs, ground_pairs = len(pred_set), len(ground_set)
    tp = 1. * len(pred_set.intersection(ground_set))
    fp = len(pred_set) - tp
    fn = len(ground_set) - tp
    tn = total - tp - fp - fn
    mcc = f1 = 0. 
    if len(pred_set) != 0 and len(ground_set) != 0:
        precision = tp / len(pred_set)
        recall = tp / len(ground_set)
        if tp > 0:
            f1 = 2 / (1/precision + 1/recall)
        if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
            mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return Scores(tp, fp, fn, tn, mcc, f1, ground_pairs, pred_pairs)
