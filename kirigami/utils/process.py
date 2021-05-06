from itertools import permutations
import torch
from kirigami._globals import *
from kirigami.utils.convert import dense2sequence


__all__ = ["binarize", "get_scores"]


def binarize(seq: torch.Tensor,
             lab: torch.Tensor,
             max_pad: int = 512,
             thres: float = .5,
             min_dist: int = 4,
             symmetrize: bool = True,
             canonicalize: bool = True) -> torch.Tensor:
    """binarizes contact matrix from deep network"""
    lab_ = lab.squeeze()
    if symmetrize:
        lab_ += lab_.T.clone()
        lab_ /= 2
    seq_length = seq.sum().item()
    seq_str = dense2sequence(seq)
    
    beg = (max_pad - seq_length) // 2
    end = beg + seq_length
     
    pairs_probs = []
    for i in range(beg, end):
        for j in range(i+min_dist, end):
            prob = lab_[i,j]
            if prob >= thres and (not canonicalize or seq_str[i-beg]+seq_str[j-beg] in CANONICAL_CHARS):
                pairs_probs.append((prob,(i,j)))

    pairs_probs.sort(reverse=True)
    out = torch.zeros((max_pad,max_pad), device=lab.device)
    dot_bracket = seq_length * ["."]
    for prob, (i, j) in pairs_probs:
        if dot_bracket[i-beg] != "." or dot_bracket[j-beg] != ".":
            continue
        out[i,j] = out[j,i] = 1.
        dot_bracket[i-beg], dot_bracket[j-beg] = "(", ")"

    while out.dim() < lab.dim():
        out.unsqueeze_(0)

    return out


def get_scores(pred_map: PairMap, ground_map: PairMap) -> Scores: 
    """returns various evaluative scores of predicted secondary structure"""
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
