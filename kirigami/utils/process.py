from itertools import permutations
import torch
from kirigami._globals import *
from kirigami.utils.convert import dense2sequence


__all__ = ["python_binarize", "get_scores"]


def python_binarize(seq: str,
                    lab: torch.Tensor,
                    max_pad: int = 512,
                    thres_pairs: float = .5,
                    thres_prob: float = 0.0,
                    min_dist: int = 4,
                    symmetrize: bool = True,
                    canonicalize: bool = True) -> torch.Tensor:
    """binarizes contact matrix from deep network"""
    lab_ = lab.squeeze()
    if symmetrize:
        lab_ += lab_.T.clone()
        lab_ /= 2
    seq_length = len(seq)
    beg = (max_pad - seq_length) // 2
    end = beg + seq_length
     
    pairs_probs = []
    for i in range(beg, end):
        for j in range(i+min_dist, end):
            prob = lab_[i,j]
            if prob >= thres_prob and (not canonicalize or seq[i-beg]+seq[j-beg] in CANONICAL_CHARS):
                pairs_probs.append((prob,(i,j)))

    pairs_probs.sort(reverse=True)
    out = torch.zeros((max_pad,max_pad), device=lab.device)
    dot_bracket = seq_length * ["."]
    max_iter = min(len(pairs_probs), thres_pairs)
    i = 0
    for prob, (j, k) in pairs_probs:
        if i == max_iter:
            break
        if dot_bracket[j-beg] != "." or dot_bracket[k-beg] != ".":
            continue
        out[j,k] = out[k,j] = 1.
        dot_bracket[j-beg], dot_bracket[k-beg] = "(", ")"
        i += 1
        
    while out.dim() < lab.dim():
        out.unsqueeze_(0)

    return out


def get_scores(pred_map: ContactMap, ground_map: ContactMap) -> Scores: 
    """returns various evaluative scores of predicted secondary structure"""
    length = len(pred_map)
    assert length == len(ground_map)
    total = length * (length-1) / 2
    pred_set = {pair for pair in pred_map.items() if pair[0] >= pair[1]}
    ground_set = {pair for pair in ground_map.items() if pair[0] >= pair[1]}
    pred_pairs, ground_pairs = len(pred_set), len(ground_set)
    tp = float(len(pred_set.intersection(ground_set)))
    fp = len(pred_set) - tp
    fn = len(ground_set) - tp
    tn = total - tp - fp - fn
    mcc = f1 = 0. 
    if len(pred_set) != 0 and len(ground_set) != 0:
        sn = tp / (tp+fn)
        pr = tp / (tp+fp)
        if tp > 0:
            f1 = 2*sn*pr / (pr+sn)
        if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
            mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return Scores(tp, tn, fp, fn, f1, mcc, ground_pairs, pred_pairs)
