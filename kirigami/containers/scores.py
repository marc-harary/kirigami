from dataclasses import dataclass

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
