import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from itertools import permutations
import torch
import munch


__all__ = ['path2munch', 'read_label', 'calcF1MCC', 'get_contacts', 'pairs2bpseq']


PairMap = DefaultDict[int,int]


def path2munch(path: Path) -> munch.Munch:
    '''Reads .json file saved at PATH and returns `Munch` object'''
    with open(path, 'r') as f:
        txt = f.read()
    conf_json = json.loads(txt)
    conf =  munch.munchify(conf_json)
    return conf


def get_contacts(input: torch.Tensor, thres: float = .5, diagonal: float = 0.) -> Tuple[torch.Tensor, PairMap]:
    '''Predicts contact matrix and index map based on output of network'''
    mat = input.squeeze()
    assert mat.dim() == 2 and mat.shape[0] == mat.shape[1], "Input tensor must be square"
    idxs = list(permutations(range(mat.shape[0]), 2))
    vals = list(zip(idxs, [input[idx] for idx in idxs]))
    vals = list(filter(lambda val: val[1] >= thres, vals))
    vals = sorted(vals, key=itemgetter(1))
    out = torch.zeros_like(mat)
    pair_dict = defaultdict(lambda: -1)
    while vals:
        val = vals.pop()
        i, j = val[0]
        out[i,j], out[j,i] = 1., 1.
        pair_dict[i], pair_dict[j] = j, i
        vals = list(filter(lambda val: not set((i,j)).intersection(set(val[0])), vals))
    out.diagonal()[:] = diagonal
    return out, pair_dict


def pairs2bpseq(sequence: str, pairs: PairMap) -> str:
    '''Turn contact matrix and sequence into bpseq file-style string'''
    out_list = [f'{i+1} {char.upper()} {pairs[i]+1}\n' for i, char in enumerate(sequence)]
    return ''.join(out_list)


def read_label(in_file: str) -> PairMap:
    '''Reads label file and returns base pairs and sequence length'''
    label_list = []
    for line in lines:
        if line.startswith('#') or line.startswith('i'):
            continue
        items = line.split()
        nt1 = int(items[0])
        nt2 = int(items[1])
        if nt1 < nt2:
            label_list.append((nt1,nt2))
        else:
            label_list.append((nt2,nt1))
    BPnat = len(label_list)
    return label_list, BPnat


def calcF1MCC(sequence: str, positive_list: PairMap, predict_list: PairMap) -> Tuple[float,float]:
    '''Returns F1 score and MCC of sequence and predicted contact points'''
    L = len(sequence)
    total = L * (L-1) / 2
    predicted = set(predict_list)
    positive = set(positive_list)
    TP = 1.*len(predicted.intersection(positive))
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
