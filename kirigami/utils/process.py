from itertools import permutations
from operator import itemgetter
import torch
from kirigami._globals import *
from kirigami.utils.convert import tensor2sequence


__all__ = ["bad_binarize", "get_scores"]


# def binarize(lab: torch.Tensor,
#              seq: torch.Tensor,
#              thres: float = .5,
#              diagonal: float = 0.,
#              min_dist: int = 4,
#              canonicalize: bool = True) -> torch.Tensor:
#     """Binarizes contact matrix from deep network"""
#     lab_ = lab.squeeze()
#     lab_flat = lab_.view(-1)
#     seq_ = seq.squeeze()
#     seq_ = seq_[:4,:,0].T.squeeze()
#     length = seq_.shape[0]
# 
#     probs, idxs = torch.sort(lab_flat, descending=True)
#     
#     hit_pairs = []
#     out = torch.zeros_like(lab_)
#     out_flat = out.view(-1)
# 
#     for prob, idx in zip(probs, idxs):
#         if prob < thres:
#             continue
#         i = int(idx / length)
#         j = idx % length
#         if abs(i - j) < min_dist or (j, i) in hit_pairs:
#             continue
#         if not canonicalize:
#             hit_pairs.append((i,j))
#             out[i,j] = out[j,i] = 1.
#         else:
#             pair_i = *(int(k) for k in seq_[i]),
#             pair_j = *(int(k) for k in seq_[j]),
#             if (pair_i, pair_j) in CANONICAL_TUPLES:
#                 hit_pairs.append((i,j))
#                 out[i,j] = out[j,i] = 1.
# 
#     out.fill_diagonal_(diagonal)
#     while out.dim() < lab.dim():
#         out.unsqueeze_(0)
# 
#     return out
# 
def bad_binarize(lab: torch.Tensor,
                 seq: torch.Tensor,
                 thres: float = .5,
                 min_dist: int = 4,
                 canonicalize: bool = True) -> torch.Tensor:
    """Binarizes contact matrix from deep network"""
    lab_ = lab.squeeze()
    seq_str = tensor2sequence(seq)
    L = len(seq_str)

    pairs_probs = []
    for i in range(L):
        for j in range(i+min_dist, L):
            prob = lab_[i,j]
            if prob >= thres and seq_str[i]+seq_str[j] in CANONICAL_CHARS:
                pairs_probs.append((prob,(i,j)))

    pairs_probs.sort(reverse=True)
    out = torch.zeros((L,L), device=lab.device)
    dot_bracket = L * ["."]
    for prob, pair in pairs_probs:
        i, j = pair
        if dot_bracket[i] != "." or dot_bracket[j] != ".":
            continue
        out[i,j] = out[j,i] = 1.
        dot_bracket[i], dot_bracket[j] = "(", ")"

    while out.dim() < lab.dim():
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
