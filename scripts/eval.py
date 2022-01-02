import os
import sys
import json
import itertools
from functools import reduce
import time
import logging
import datetime
from pathlib import Path
from tqdm import tqdm
from munch import munchify, Munch
from dataclasses import dataclass, asdict
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn.functional import binary_cross_entropy

import kirigami.nn
from kirigami.nn.utils import *
from kirigami.utils import binarize


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


def to_str(ipt):
    ipt_ = ipt.squeeze()
    total_length = ipt_.shape[1]
    fasta_length = int(ipt_.sum().item())
    beg = (total_length - fasta_length) // 2
    end = beg + fasta_length
    _, js = torch.max(ipt_[:,beg:end], 0)
    return "".join("AUCG"[j] for j in js)


# def get_scores(pred, grd, length):
#     total = length * (length-1) / 2
# 
#     total_length = pred.shape[0]
#     beg = (total_length - length) // 2
#     end = beg + length
# 
#     vals, idxs = torch.max(pred[:,beg:end], 0)
#     prd_pairs = set()
#     for i, (val, idx) in enumerate(zip(vals, idxs)):
#         if val == 1:
#             prd_pairs.add((i, idx))
# 
#     vals, idxs = torch.max(grd[:,beg:end], 0)
#     grd_pairs = set()
#     for i, (val, idx) in enumerate(zip(vals, idxs)):
#         if val == 1:
#             grd_pairs.add((i, idx))
# 
#     n_prd, n_grd = len(prd_pairs), len(grd_pairs)
#     tp = float(len(prd_pairs.intersection(grd_pairs)))
#     fp = len(prd_pairs) - tp
#     fn = len(grd_pairs) - tp
#     tn = total - tp - fp - fn
#     mcc = f1 = 0. 
#     if n_prd > 0 and n_grd > 0:
#         sn = tp / (tp+fn)
#         pr = tp / (tp+fp)
#         if tp > 0:
#             f1 = 2*sn*pr / (pr+sn)
#         if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
#             mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
#     return Scores(tp, tn, fp, fn, f1, mcc, n_grd, n_prd)


def get_scores(pred, grd):
    length = pred.shape[0]

    total = length * (length-1) / 2

    vals, idxs = torch.max(pred, 0)
    prd_pairs = set()
    for i, (val, idx) in enumerate(zip(vals, idxs)):
        if val == 1:
            prd_pairs.add((i, idx.item()))

    vals, idxs = torch.max(grd, 0)
    grd_pairs = set()
    for i, (val, idx) in enumerate(zip(vals, idxs)):
        if val == 1:
            grd_pairs.add((i, idx.item()))

    # print("PREDICTED")
    # for pair in prd_pairs:
    #     print(pair)
    # 
    # print("GROUND")
    # for pair in grd_pairs:
    #     print(pair)

    # print(5*"\n")

    n_prd, n_grd = len(prd_pairs), len(grd_pairs)
    tp = float(len(prd_pairs.intersection(grd_pairs)))
    fp = len(prd_pairs) - tp
    fn = len(grd_pairs) - tp
    tn = total - tp - fp - fn
    mcc = f1 = 0. 
    if n_prd > 0 and n_grd > 0:
        sn = tp / (tp+fn)
        pr = tp / (tp+fp)
        if tp > 0:
            f1 = 2*sn*pr / (pr+sn)
        if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
            mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return Scores(tp, tn, fp, fn, f1, mcc, n_grd, n_prd)


def main():
    with open(sys.argv[1], "r") as f:
        txt = f.read()
        p = munchify(json.loads(txt))

    # load data and copy into `DataLoader`'s
    vl_set = torch.load(p.vl_set)
    crit = eval(p.criterion)()

    stats = []

    for i, batch in enumerate(tqdm(vl_set)):
        fasta, thermo, con_grd = batch
        thermo = thermo.to_dense()
        con_grd = con_grd.to_dense().float()
        fasta2d = fasta.to_dense()

        thres_pairs = int(con_grd.sum().item() / 2)
        seq = to_str(fasta2d)

        beg = (fasta2d.shape[1] - len(seq)) // 2
        end = beg + len(seq)

        bin_pred = binarize(lab=thermo,
                            seq=seq,
                            min_dist=4,
                            thres_pairs=thres_pairs,
                            thres_prob=0,
                            symmetrize=False,
                            canonicalize=False)


        if isinstance(crit, torch.nn.BCEWithLogitsLoss):
            bin_loss = binary_cross_entropy(bin_pred[beg:end, beg:end], con_grd[beg:end, beg:end])
        else:
            bin_loss = crit(bin_pred, con_grd)

        scores = get_scores(bin_pred[beg:end, beg:end], con_grd[beg:end, beg:end])
        stats.append(asdict(scores))

    df = pd.DataFrame(stats) 
    df.to_csv("thermo_scores.csv") 



if __name__ == "__main__":
    main()
