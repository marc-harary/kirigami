from itertools import permutations
from operator import itemgetter
import torch
from kirigami._globals import *


__all__ = ["binarize", "get_scores"]


def binarize(contact: torch.Tensor,
             sequence: torch.Tensor,
             thres: float = .5,
             diagonal: float = 0.,
             min_dist: int = 4,
             canonicalize: bool = True) -> torch.Tensor:
    """Binarizes contact matrix from deep network"""
    contact_squeeze = contact.squeeze()
    sequence_squeeze = sequence.squeeze()
    sequence_squeeze = sequence_squeeze[:4,:,0].T.squeeze()
    assert contact_squeeze.dim() == sequence_squeeze.dim() == 2
    assert contact_squeeze.shape[0] == contact_squeeze.shape[1] == sequence_squeeze.shape[0] 

    out = torch.zeros_like(contact_squeeze)

    pairs = list(permutations(range(contact_squeeze.shape[0]), 2))
    probs = [contact_squeeze[pair] for pair in pairs]
    pairs_probs = list(zip(pairs, probs))
    pairs_probs.sort(key=itemgetter(1))

    hit_pairs = []
    for pair, prob in pairs_probs:
        i, j = pair
        pairs = tuple(sequence_squeeze[i].tolist()), tuple(sequence_squeeze[j].tolist())
        if abs(i - j) >= min_dist and \
           prob >= thres and \
           (not canonicalize or pairs in CANONICALS_TENSOR) and (j, i) not in hit_pairs:
            hit_pairs.append(pair)
            out[i,j] = out[j,i] = 1.

    out.fill_diagonal_(diagonal)
    while out.dim() < contact.dim():
        out.unsqueeze_(0)

    return out


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
