import string
import os
import subprocess
import re
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import (binary_matthews_corrcoef,
    binary_f1_score, binary_precision, binary_recall)
# import RNA

import pyximport
# pyximport.install(setup_args=dict(include_dirs=np.get_include()))
# import nussinov

# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location("_RNA", "/ysm-gpfs/apps/software/ViennaRNA/2.4.11-foss-2018b-Python-3.7.0/lib/python3.7/site-packages/_RNA.cpython-37m-x86_64-linux-gnu.so")
# _RNA = importlib.util.module_from_spec(spec)
# sys.modules["_RNA"] = _RNA
# spec.loader.exec_module(_RNA)
# spec = importlib.util.spec_from_file_location("RNA", "/ysm-gpfs/apps/software/ViennaRNA/2.4.11-foss-2018b-Python-3.7.0/lib/python3.7/site-packages/RNA/__init__.py")
# RNA = importlib.util.module_from_spec(spec)
# sys.modules["RNA"] = RNA
# spec.loader.exec_module(RNA)


PSEUDO_LEFT = "({[<" + string.ascii_uppercase
PSEUDO_RIGHT = ")}]>" + string.ascii_lowercase
COLORS = dict(A="red", C="green", G="blue", U="yellow")


def apply_cons(dbn, seq_str):
    CANON = {"AU", "UA", "CG", "GC", "UG", "GU"}
    idxs = parsedbn(dbn, return_idxs=True)
    idxs_kept = []
    for pair in idxs:
        i, j = min(pair), max(pair)
        if seq_str[i] + seq_str[j] in CANON and j - i >= 4:
            idxs_kept.append((i, j))
    out_dbn = dict2db(idxs_kept, len(seq_str))
    return out_dbn, parsedbn(out_dbn)


def build_table(pair_dict, L):
    memo = np.zeros((L, L))
    for k in range(1, L):
        for i in range(L - k):
            j = i + k
            unpairi = memo[i+1, j]
            unpairj = memo[i, j-1]
            pairij = memo[i+1, j-1] + int((i, j) in pair_dict or (j, i) in pair_dict)
            bifurc = 0
            for l in range(i, j):
                bifurc = max(bifurc, memo[i, l] + memo[l+1, j])
            memo[i, j] = max(unpairi, unpairj, pairij, bifurc)
    return memo


def trace_table(memo, pairs = None, i = None, j = None):
    if pairs is None:
        pairs = {}
        L = memo.shape[0]
        return trace_table(memo, pairs, 0, L-1)
    elif i >= j:
        return pairs
    elif memo[i, j] == memo[i+1, j]:
        return trace_table(memo, pairs, i+1, j)
    elif memo[i, j] == memo[i, j-1]:
        return trace_table(memo, pairs, i, j-1)
    elif memo[i, j] == memo[i+1, j-1] + 1: # canon(seq[i] + seq[j]):
        pairs[i] = j
        pairs[j] = i
        return trace_table(memo, pairs, i+1, j-1)
    else:
        for k in range(i+1, j-1):
            if memo[i, j] == memo[i, k] + memo[k+1, j]:
                trace_table(memo, pairs, i, k)
                trace_table(memo, pairs, k+1, j)
                break
    return pairs


def parsedbn(dbn, return_idxs=False, split_pseudo=False):
    stacks = tuple((deque() for _ in PSEUDO_LEFT))
    nest_idxs = []
    pseudo_idxs = []
    for i, char in enumerate(dbn):
        if char in PSEUDO_LEFT:
            stacks[PSEUDO_LEFT.index(char)].append(i)
        elif char in PSEUDO_RIGHT:
            try:
                j = stacks[PSEUDO_RIGHT.index(char)].pop()
                if char == ")":
                    nest_idxs.append((i,j))
                    nest_idxs.append((j,i))
                else:
                    pseudo_idxs.append((i,j))
                    pseudo_idxs.append((j,i))
            except IndexError:
                continue
    if return_idxs:
        return nest_idxs, pseudo_idxs
    else:
        out = torch.zeros(len(dbn), len(dbn))
        for i, j in nest_idxs + pseudo_idxs:
            out[i, j] = 1
        return out.int()


def dict2db(idxs, L):
    pair_hierarch = []
    pairs = set(idxs)
    while pairs:
        memo = build_table(pairs, L)
        pairs_nest = trace_table(memo)
        pairs_nest = set(pairs_nest.items())
        pairs -= pairs_nest
        pair_hierarch.append(pairs_nest)
    out_dbn = L * ["."]
    for left, right, pair_set in zip(PSEUDO_LEFT, PSEUDO_RIGHT, pair_hierarch):
        for pair in pair_set:
            i, j = min(pair), max(pair)
            out_dbn[i] = left
            out_dbn[j] = right
    return "".join(out_dbn)


def mat2db(mat):
    mat_ = mat.squeeze()
    L = mat_.shape[0]
    ii, jj = torch.where(mat_)
    ii, jj = ii.tolist(), jj.tolist()
    idxs = list(zip(ii, jj))
    return dict2db(idxs, L)


def embed_bpseq(path, ex_noncanon=False, ex_sharp=False):
    BASES = dict(A=0, C=1, G=2, U=3)
    CANON = {"AU", "UA", "CG", "GC", "UG", "GU"}
    with open(path, "r") as f:
        lines = f.read().splitlines()
    seq = ""
    second = {}
    line_idx = 0
    con_idxs = []
    for line in lines:
        if line.startswith("#"):
            continue
        words = line.split()
        base = words[1]
        seq += base
    for line in lines:
        if line.startswith("#"):
            continue
        words = line.split()
        ii, jj = int(words[0])-1, int(words[-1])-1
        if ii < jj:
            pair = seq[ii] + seq[jj]
            if ex_noncanon and pair not in CANON:
                continue
            if ex_sharp and abs(ii - jj) < 4:
                continue
            con_idxs.append((ii, jj))
            con_idxs.append((jj, ii))
    # fasta = torch.zeros(4, len(seq), dtype=torch.uint8)
    # idxs = [BASES[char] for char in seq]
    # fasta[idxs, list(range(len(seq)))] = 1
    dbn = dict2db(con_idxs, len(seq))
    con = torch.zeros(len(seq), len(seq), dtype=torch.uint8)
    if len(con_idxs) > 0:
        con_idxs_ = torch.tensor(con_idxs)# + offset
        con[con_idxs_[:,0], con_idxs_[:,1]] = 1
        con[con_idxs_[:,1], con_idxs_[:,0]] = 1
    return con, seq, dbn


def read_fasta(path):
    with open(path) as f:
        fasta = f.read().splitlines()[-1] 
    return fasta


def embed_fasta(fasta):
    base_dict = dict(A=torch.tensor([1,0,0,0]),
                     C=torch.tensor([0,1,0,0]),
                     G=torch.tensor([0,0,1,0]),
                     U=torch.tensor([0,0,0,1]),
                     N=torch.tensor([.25,.25,.25,.25]),
                     D=torch.tensor([1/3,0,1/3,1/3]),
                     W=torch.tensor([0.5,0,0,0.5]),
                     V=torch.tensor([1/3,1/3,1/3,0]),
                     K=torch.tensor([0,0,0.5,0.5]),
                     R=torch.tensor([.5,0,.5,0]),
                     M=torch.tensor([.5,.5,0,0]),
                     S=torch.tensor([0,.5,.5,0]),
                     Y=torch.tensor([0,.5,0,.5]))
    opt = torch.stack([base_dict[char] for char in fasta], axis=1)
    return opt


def outer_concat(fasta):
    out = fasta.unsqueeze(-1)
    out = torch.cat(out.shape[-2] * [out], dim=-1)
    out_t = out.transpose(-1, -2)
    out = torch.cat([out, out_t], dim=-3)
    out = out.unsqueeze(0)
    return out


def get_vienna_rna(fasta):
    fc = RNA.fold_compound(fasta)    
    fc.pf()
    bpp = torch.tensor(fc.bpp())
    bpp = bpp[:-1, :-1]
    return bpp


def get_con_metrics(prd, grd, threshold):
    mask = ~grd.isnan()
    grd_flat = grd[mask].int()
    prd_flat = prd[mask]
    return dict(mcc=binary_matthews_corrcoef(prd_flat, grd_flat, threshold).item(),
                f1=binary_f1_score(prd_flat, grd_flat, threshold).item(),
                precision=binary_precision(prd_flat, grd_flat, threshold).item(),
                recall=binary_recall(prd_flat, grd_flat, threshold).item())

def embed_st(path):
    name = ""
    fasta = ""
    dbn = ""
    with open(path, "r") as f:
        line = f.readline()
        name = line.split()[-1]
        while line.startswith("#"):
            line = f.readline()
        fasta = line.upper().strip()
        dbn = f.readline().strip()
    con = parsedbn(dbn)
    fasta = embed_fasta(fasta)  
    return fasta, con

