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
