from typing import Set, Tuple
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


def build_table(pair_set: Set[Tuple[int, int]], length: int) -> np.ndarray:
    """
    Creates memoization table for Nussinov's dynamic programming algorithm.

    Parameters
    ----------
    pair_dict : set
        Set of ordered pairs of paired indices in solution.
    length : int
        Total length of molecule.

    Returns
    -------
    memo : np.ndarray of shape (length, length)
        Memoization table.
    """
    memo = np.zeros((length, length))
    for k in range(1, length):
        for i in range(length - k):
            j = i + k
            unpairi = memo[i + 1, j]
            unpairj = memo[i, j - 1]
            pairij = memo[i + 1, j - 1] + int((i, j) in pair_set or (j, i) in pair_set)
            bifurc = 0
            for l in range(i, j):
                bifurc = max(bifurc, memo[i, l] + memo[l + 1, j])
            memo[i, j] = max(unpairi, unpairj, pairij, bifurc)
    return memo


def trace_table(
    memo,
    pairs: Optional[Dict[int, int]] = None,
    i: int = None,
    j: int = None,
) -> Dict[int, int]:
    """
    Traces memoization table for Nussinov's algorithm.

    Parameters
    ----------
    memo : np.ndarray
        Memoization table for dynamic programming.
    pairs : dict
        Dictionary of pairs secondary structure solution to be updated by
        current recursive call of `trace_table`.
    i : int
        Left residue index in current recursive call of `trace_table`.
    j : int
        Right residue index in current recursive call of `trace_table`.

    Returns
    -------
    pairs : dict
        Dictionary of pairs secondary structure solution to be updated by
        current recursive call of `trace_table`.
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


def parsedbn(dbn: str, return_idxs: bool = False) -> torch.Tensor:
    """
    Converts a dot-bracket string to adjacency matrix as a torch.Tensor.

    Parameters
    ----------
    dbn : str of length (n,)
        Dot-bracket string to conver to tensor.
    return_idxs : bool
        If `True`, return the indices for each level of pseudoknots.

    Returns
    -------
    out : torch.Tensor of shape (n, n)
        Adjacency matrix.

    Notes
    -----
    Does NOT perform checks for validity of input string.
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


def dict2db(idxs: Dict[int, int], length: int) -> str:
    """
    Converts a dictionary of pairs to a dot-bracket string by iteratively
    applying Nussinov's algorithm to identify successive levels in the
    hierarchy of pseudoknots.

    Parameters
    ----------
    idxs : dict
        Dictionary representing pairs in secondary structure.
    length : int
        Total length of molecule.

    Returns
    -------
    out : str
        Dot-bracket string.
    """
    pair_hierarch = []
    pairs = set(idxs)
    while pairs:
        memo = build_table(pairs, length)
        pairs_nest = trace_table(memo)
        pairs_nest = set(pairs_nest.items())
        pairs -= pairs_nest
        pair_hierarch.append(pairs_nest)
    out_list = length * ["."]
    for left, right, pair_set in zip(PSEUDO_LEFT, PSEUDO_RIGHT, pair_hierarch):
        for pair in pair_set:
            i, j = min(pair), max(pair)
            out_list[i] = left
            out_list[j] = right
    out = "".join(out_list)
    return out


def mat2db(mat: torch.Tensor) -> str:
    """
    Converts adjacency matrix to a dot-bracket string.

    Parameters
    ----------
    mat : torch.Tensor of shape (n, n)
        Input adjacency matrix.

    Returns
    ------
    out : str of length n
        Output dot-bracket string.
    """
    mat_ = mat.squeeze()
    length = mat_.shape[0]
    idxi, idxj = torch.where(mat_)
    idxi, idxj = idxi.tolist(), idxj.tolist()
    idxs = list(zip(idxi, idxj))
    out = dict2db(idxs, length)
    return out


def embed_fasta(path: str) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """
    Reads and embeds FASTA file as `torch.Tensor`.

    Parameters
    ----------
    path : str
        Path to FASTA file.

    Returns
    -------
    mols: list
        List of molecule names in FASTA file.
    fasta_strs : list
        List of primary sequences in FASTA file.
    fasta_embeds : list
        List of primary sequences embedded as `torch.Tensor`s.
    """
    mols, fasta_strs, fasta_embeds = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i in tqdm(range(len(lines) // 2)):
        mols.append(lines[2 * i][1:].strip())
        fasta_strs.append((fasta_str := lines[2 * i + 1].strip().upper()))
        fasta_embeds.append((_embed_fasta(fasta_str),))
    return mols, fasta_strs, fasta_embeds


def _embed_fasta(fasta: str) -> torch.Tensor:
    """
    Embeds FASTA string as `torch.Tensor`.

    Parameters
    ----------
    fasta : str of length n
        Primary sequence string.

    Returns
    -------
    opt : torch.Tensor of shape (4, n)
        One-hot encoded sequence.

    Notes
    -----
    Degenerate bases are represented by averages of corresponding one-hot
    encoded nucleotides.
    """
    opt = torch.stack([BASE_DICT[char] for char in fasta], axis=1)
    return opt


def embed_dbn(path: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Embeds DBN file as list of `torch.Tensor`s corresponding to embedded
    primary and secondary structures.

    Parameters
    ----------
    path : str
        Path to DBN file.

    Returns
    -------
    out : list
        List of ordered pairs of `torch.Tensors` corresponding to embedded
        primary and secondary structures.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i in tqdm(range(len(lines) // 3)):
        fasta_str = lines[3 * i + 1].upper()
        dbn = lines[3 * i + 2]
        fasta_embed = _embed_fasta(fasta_str)
        dbn_embed = parsedbn(dbn)
        out.append((fasta_embed, dbn_embed))
    return out


def get_con_metrics(
    prd: torch.Tensor,
    grd: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """
    Computes MCC, F1 score, precision, and recall metrics of predicted
    adjacency matrix.

    Parameters
    ----------
    prd : torch.Tensor of shape (n,n)
        Predicted tensor.
    grd : torch.Tensor of shape (n,n)
        Ground-truth tensor.
    threshold: float
        Threshold for classification metrics.

    Returns
    -------
    out : dict
        Dictionary of classification metrics.
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
