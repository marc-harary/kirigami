'''utils for converting and embedding various data types'''


import json
import re
from pathlib import Path
from typing import Tuple
from collections import defaultdict
from operator import itemgetter
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
           'tensor2sequence',
           'tensor2bpseq',
           'bpseq2pairmap',
           'pairmap2bpseq',
           'binarize',
           'get_scores']


def path2munch(path: Path) -> munch.Munch:
    '''Reads .json file saved at PATH and returns `Munch` object'''
    with open(path, 'r') as f:
        txt = f.read()
    conf_json = json.loads(txt)
    conf =  munch.munchify(conf_json)
    return conf


def pairmap2tensor(pairs: PairMap, out_dim: int = 3) -> torch.Tensor:
    '''Converts `PairMap` to contact matrix (`torch.Tensor`)'''
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
    '''Converts `FASTA` sequence to `torch.Tensor`'''
    sequence_copy = sequence.strip().upper()
    length = len(sequence_copy)
    one_hot = torch.stack([BASE_DICT[char] for char in sequence_copy])
    out = torch.empty(2 * N_BASES, length, length)
    for i in range(length):
        for j in range(length):
            out[:,i,j] = torch.cat((one_hot[i], one_hot[j]))
    return out


def label2tensor(label: str, out_dim: int = 4) -> torch.Tensor:
    '''Converts label file to contact matrix (`torch.Tensor`)'''
    lines = label.splitlines()
    matches = re.findall(r'[\d]+$', lines[0])
    length = int(matches[0])
    out = torch.zeros(length, length)
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
    length = len(lines)
    pair_default = defaultdict(lambda: NO_CONTACT)
    sequence = ''
    for line in lines:
        i, base, j = line.split()
        i, j = int(i) - 1, int(j) - 1
        pair_default[i], pair_default[j] = j, i
        sequence += base.upper()
    pair_map = {i: pair_default[i] for i in range(length)}
    return sequence, pair_map


def bpseq2tensor(bpseq: str) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Converts `.bpseq` file to `torch.Tensor`'s'''
    sequence, pair_map = bpseq2pairmap(bpseq)
    return sequence2tensor(sequence), pairmap2tensor(pair_map)


def pairmap2bpseq(sequence: str, pair_map: PairMap) -> str:
    '''Converts `FASTA`-style sequence and `PairMap` to `.bpseq`-style string'''
    assert len(sequence) == len(pair_map)
    out_list = [f'{i+1} {char.upper()} {pair_map[i]+1}\n' for i, char in enumerate(sequence)]
    return ''.join(out_list)


def binarize(ipt: torch.Tensor, thres: float = .5, diagonal: float = 0.) -> torch.Tensor:
    '''Binarizes contact matrix from deep network'''
    mat = ipt.squeeze()
    length = mat.shape[0]
    assert mat.dim() == 2 and length == mat.shape[1], "Input tensor must be square"
    idxs = list(permutations(range(length), 2))
    vals = list(zip(idxs, [mat[idx] for idx in idxs]))
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


def tensor2pairmap(ipt: torch.Tensor) -> PairMap:
    '''Converts binarized contact matrix to `PairMap`'''
    mat = ipt.squeeze()
    assert mat.dim() == 2
    values, js = torch.max(mat, 1)
    js[values == 0.] = NO_CONTACT
    js_ints = map(int, js)
    pair_map = dict(enumerate(js_ints))
    return pair_map


def tensor2sequence(ipt: torch.Tensor) -> str:
    '''Converts embedded `FASTA` sequence to string'''
    chars_embed = ipt.squeeze()
    chars_embed = chars_embed[:N_BASES, :, 0].T
    chars = []
    for row in chars_embed:
        _, idx = torch.max(row, 0)
        chars.append(BASES[idx])
    return ''.join(chars)


def tensor2bpseq(sequence: torch.Tensor, label: torch.Tensor) -> str:
    '''Converts sequence and label tensors to `bpseq`-like string'''
    sequence_str = tensor2sequence(sequence)
    label_pair_map = tensor2pairmap(label)
    return pairmap2bpseq(sequence_str, label_pair_map)


def get_scores(pred_map: PairMap, ground_map: PairMap) -> Dict[str,float]:
    '''Returns various evaluative scores of predicted secondary structure'''
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
    out = dict(tp=tp, fp=fp, fn=fn, tn=tn, pred_pairs=pred_pairs, ground_pairs=ground_pairs)
    out['mcc'] = out['f1'] = 0
    if len(pred_set) != 0 and len(ground_set) != 0:
        precision = tp / len(pred_set)
        recall = tp / len(ground_set)
        if tp > 0:
            out['f1'] = 2 / (1/precision + 1/recall)
        if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
            out['mcc'] = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return out
