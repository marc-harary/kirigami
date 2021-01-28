import json
import os
import pathlib
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from multipledispatch import dispatch
import munch
import torch


__all__ = ['path2munch', 'read_label', 'calcF1MCC', 'get_contacts']


PairMap = DefaultDict[int,int]


def get_idxs(input: torch.Tensor, skip_diagonal: bool = False) -> PairMap:
    '''Returns all permutations of indices in a matrix'''
    L = input.shape[0]
    idxs = []
    for i in range(L):
        for j in range(L):
            if skip_diagonal and i == j:
                continue
            idxs.append((i,j))
    return idxs


def get_contacts(input: torch.Tensor, thres: float = .5) -> Tuple[torch.Tensor, PairMap]
    '''Predicts contact matrix based on output of network'''
    mat = input.squeeze()
    assert mat.dim() == 2 and mat.shape[0] == mat.shape[1], "Input tensor must be square"
    idxs = get_idxs(mat, skip_diagonal=True)
    vals = list(zip(idxs, [input[idx] for idx in idxs]))
    vals = list(filter(lambda val: val[1] >= thres, vals))
    vals = sorted(vals, key=itemgetter(1), reverse=True)
    pairs = set()
    out = torch.zeros_like(mat)
    pair_dict = defaultdict(lambda: 0)
    for pair in vals:
        idxs = pair[0]
        idxs_sort = tuple(sorted(idxs))
        idxs_rev = tuple(reversed(idxs))
        if idxs_sort not in pairs:
            pairs.add(idxs_sort)
            out[idxs] = 1.
            out[idxs_rev] = 1.
            pair_dict[idxs[0]] = idxs[1]
            pair_dict[idxs[1]] = idxs[0]
    out.diagonal()[:] = 1.
    return out, pair_list


def pairs2bpseq(sequence: str, pairs: PairMap) -> str:
    '''Turn contact matrix and sequence into bpseq file-style string'''
    out_str = ''
    for i, char in enumerate(sequence):
        out_str += f'f{i+1} {char} {pairs[i]+1}\n'
    return out_str


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
