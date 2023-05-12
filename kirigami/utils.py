from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from torchmetrics.functional.classification import (
    binary_matthews_corrcoef,
    binary_f1_score,
    binary_precision,
    binary_recall,
)
from kirigami.constants import PSEUDO_LEFT, PSEUDO_RIGHT, BASE_DICT


def build_table(pair_dict, length):
    """
    Creates memoization table for Nussinov's algorithm.
    """
    memo = np.zeros((length, length))
    for k in range(1, length):
        for i in range(length - k):
            j = i + k
            unpairi = memo[i + 1, j]
            unpairj = memo[i, j - 1]
            pairij = memo[i + 1, j - 1] + int(
                (i, j) in pair_dict or (j, i) in pair_dict
            )
            bifurc = 0
            for l in range(i, j):
                bifurc = max(bifurc, memo[i, l] + memo[l + 1, j])
            memo[i, j] = max(unpairi, unpairj, pairij, bifurc)
    return memo


def trace_table(memo, pairs=None, i=None, j=None):
    """
    Traces memoization table for Nussinov's algorithm.
    """
    if pairs is None:
        pairs = {}
        length = memo.shape[0]
        return trace_table(memo, pairs, 0, length - 1)
    if i >= j:
        return pairs
    if memo[i, j] == memo[i + 1, j]:
        return trace_table(memo, pairs, i + 1, j)
    if memo[i, j] == memo[i, j - 1]:
        return trace_table(memo, pairs, i, j - 1)
    if memo[i, j] == memo[i + 1, j - 1] + 1:  # canon(seq[i] + seq[j]):
        pairs[i] = j
        pairs[j] = i
        return trace_table(memo, pairs, i + 1, j - 1)
    for k in range(i + 1, j - 1):
        if memo[i, j] == memo[i, k] + memo[k + 1, j]:
            trace_table(memo, pairs, i, k)
            trace_table(memo, pairs, k + 1, j)
            break
    return pairs


def parsedbn(dbn, return_idxs=False):
    """
    Converts a dot-bracket string to torch.Tensor.
    """
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
                    nest_idxs.append((i, j))
                    nest_idxs.append((j, i))
                else:
                    pseudo_idxs.append((i, j))
                    pseudo_idxs.append((j, i))
            except IndexError:
                continue
    if return_idxs:
        return nest_idxs, pseudo_idxs
    out = torch.zeros(len(dbn), len(dbn))
    for i, j in nest_idxs + pseudo_idxs:
        out[i, j] = 1
    return out.int()


def dict2db(idxs, length):
    """
    Converts a dictionary of pairs to a dot-bracket string.
    """
    pair_hierarch = []
    pairs = set(idxs)
    while pairs:
        memo = build_table(pairs, length)
        pairs_nest = trace_table(memo)
        pairs_nest = set(pairs_nest.items())
        pairs -= pairs_nest
        pair_hierarch.append(pairs_nest)
    out_dbn = length * ["."]
    for left, right, pair_set in zip(PSEUDO_LEFT, PSEUDO_RIGHT, pair_hierarch):
        for pair in pair_set:
            i, j = min(pair), max(pair)
            out_dbn[i] = left
            out_dbn[j] = right
    return "".join(out_dbn)


def mat2db(mat):
    """
    Converts a torch.Tensor to a dot-bracket string.
    """
    mat_ = mat.squeeze()
    length = mat_.shape[0]
    idxi, idxj = torch.where(mat_)
    idxi, idxj = idxi.tolist(), idxj.tolist()
    idxs = list(zip(idxi, idxj))
    return dict2db(idxs, length)


def embed_fasta(path):
    """
    Reads and embeds FASTA file as torch.Tensor.
    """
    mols, fasta_strs, fasta_embeds = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i in tqdm(range(len(lines) // 2)):
        mols.append(lines[2 * i][1:].strip())
        fasta_strs.append((fasta_str := lines[2 * i + 1].strip().upper()))
        fasta_embeds.append((_embed_fasta(fasta_str),))
    return mols, fasta_strs, fasta_embeds


def _embed_fasta(fasta):
    """
    Embeds FASTA string as torch.Tensor.
    """
    opt = torch.stack([BASE_DICT[char] for char in fasta], axis=1)
    return opt


def embed_dbn(path):
    """
    Embeds dot-bracket file as torch.Tensor's.
    """
    opt = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i in tqdm(range(len(lines) // 3)):
        fasta_str = lines[3 * i + 1].upper()
        dbn = lines[3 * i + 2]
        fasta_embed = _embed_fasta(fasta_str)
        dbn_embed = parsedbn(dbn)
        opt.append((fasta_embed, dbn_embed))
    return opt


def get_con_metrics(prd, grd, threshold):
    """
    Evaluates classification metrics of predicted adjacency matrix.
    """
    idxs = torch.ones_like(prd.squeeze(), dtype=bool).triu(1)
    grd_flat = grd.squeeze()[idxs].int()
    prd_flat = prd.squeeze()[idxs]
    return dict(
        mcc=binary_matthews_corrcoef(prd_flat, grd_flat, threshold).item(),
        f1=binary_f1_score(prd_flat, grd_flat, threshold).item(),
        precision=binary_precision(prd_flat, grd_flat, threshold).item(),
        recall=binary_recall(prd_flat, grd_flat, threshold).item(),
    )
