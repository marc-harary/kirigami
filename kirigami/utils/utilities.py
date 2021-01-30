import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from itertools import permutations
import torch
import munch
from kirigami._globals import *


__all__ = ['path2munch',
           'pairmap2tensor',
           'sequence2tensor',
           'label2tensor',
           'bpseq2tensor',
           'tensor2pairmap',
           'bpseq2pairmap',
           'pairmap2bpseq',
           'binarize',
           'calcF1MCC']


def path2munch(path: Path) -> munch.Munch:
    '''Reads .json file saved at PATH and returns `Munch` object'''
    with open(path, 'r') as f:
        txt = f.read()
    conf_json = json.loads(txt)
    conf =  munch.munchify(conf_json)
    return conf


def pairmap2tensor(pairs: PairMap, out_dim: int = 4) -> torch.Tensor:
    '''Converts `PairMap` to contact matrix (`torch.Tensor`)'''
    L = len(pairs)
    out = torch.zeros(L, L)
    for i, j in pairs.items():
        if j == -1:
            continue
        out[i,j] = 1.
    while out_dim > out.dim():
       out.unsqueeze_(0)
    return out


def sequence2tensor(sequence: str) -> torch.Tensor:
    '''Converts `FASTA` sequence to `torch.Tensor`'''
    L = len(sequence)
    one_hot = torch.stack([BASE_DICT[char] for char in sequence.upper()])
    out = torch.empty(2 * N_BASES, L, L)
    for i in range(L):
        for j in range(L):
            out[:,i,j] = torch.cat((one_hot[i], one_hot[j]))
    return out


def label2tensor(label: str, out_dim: int = 4) -> torch.Tensor:
    '''Converts label file to contact matrix (`torch.Tensor`)'''
    lines = label.splitlines()
    matches = re.findall(r'[\d]+$', lines[0])
    L = int(matches[0])
    out = torch.zeros(L, L)
    for line in lines:
        if line.startswith('#') or line.startswith('i'):
            continue
        line_split = line.split()
        idx1, idx2 = int(line_split[0]), int(line_split[-1])
        out[idx1-1, idx2-1] = 1.
    while out_dim > out.dim():
        out.unsqueeze_(0)
    return out


def bpseq2pairmap(bpseq: str) -> Tuple[str, PairMap]:
    '''Converts `.bpseq` file to string and `PairMap`'''
    lines = bpseq.splitlines()
    lines = list(filter(lambda line: not line.startswith('#'), lines))
    L = len(lines)
    pair_default = defaultdict(lambda: NO_CONTACT)
    sequence = ''
    for line in lines:
        i, base, j = line.split()
        i, j = int(i) - 1, int(j) - 1
        pair_default[i], pair_default[j] = j, i
        sequence += base.upper()
    pair_map = {i: pair_default[i] for i in range(L)}
    return sequence, pair_map


def bpseq2tensor(bpseq: str) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Converts `.bpseq` file to `torch.Tensor`'s'''
    sequence, pair_map = bpseq2pairmap(bpseq)
    return sequence2tensor(sequence), pairmap2tensor(pair_map)


def pairmap2bpseq(sequence: str, pair_map: PairMap) -> str:
    '''Converts `FASTA`-style sequence and `PairMap` to `.bpseq`-style string'''
    assert len(sequence) == len(pairs)
    out_list = [f'{i+1} {char.upper()} {pair_map[i]+1}\n' for i, char in enumerate(sequence)]
    return ''.join(out_list)


def binarize(input: torch.Tensor, thres: float = .5, diagonal: float = 0.) -> torch.Tensor:
    '''Binarizes contact matrix from deep network'''
    mat = input.squeeze()
    L = mat.shape[0]
    assert mat.dim() == 2 and L == mat.shape[1], "Input tensor must be square"
    idxs = list(permutations(L, 2))
    vals = list(zip(idxs, [input[idx] for idx in idxs]))
    vals = list(filter(lambda val: val[1] >= thres, vals))
    vals = sorted(vals, key=itemgetter(1))
    out = torch.zeros_like(mat)
    while vals:
        val = vals.pop()
        i, j = val[0]
        out[i,j], out[j,i] = 1., 1.
        vals = list(filter(lambda val: not set((i,j)).intersection(set(val[0])), vals))
    out.fill_diagonal_(diagonal)
    return out


def tensor2pairmap(input: torch.Tensor) -> PairMap:
    '''Converts binarized contact matrix to `PairMap`'''
    assert input.dim() == 2
    values, js = torch.max(input, 1)
    js[values == 0.] = NO_CONTACT
    pair_map = dict(enumerate(js))
    return pair_map


def calcF1MCC(sequence: str, positive_list: PairMap, predict_list: PairMap) -> Tuple[float,float]:
    '''Returns F1 score and MCC of sequence and predicted contact points'''
    L = len(sequence)
    total = L * (L-1) / 2
    predicted = set(predict_list)
    positive = set(positive_list)
    TP = 1. * len(predicted.intersection(positive))
    FP = len(predicted) - TP
    FN = len(positive) - TP
    TN = total - TP - FP - FN
    if len(predicted) == 0 or len(positive) == 0:
        return 0, 0
    precision = TP / len(predicted)
    recall = TP / len(positive)
    F1 = 0
    MCC = 0
    if TP > 0:
        F1 = 2 / (1/precision + 1/recall)
    if (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) > 0:
        MCC = (TP*TN-FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5
    return F1, MCC
