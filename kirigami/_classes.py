from typing import OrderedDict, Tuple
from dataclasses import dataclass


__all__ = ["ContactMap",
           "Scores",
           "SeqLab",
           "Dist",
           "DistMap"]


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

ContactMap = OrderedDict[int,int]

@dataclass
class SeqLab:
    length: int
    sequence: str
    pair_map: ContactMap

@dataclass
class Dist:
    PP: float
    O5O5: float 
    C5C5: float
    C4C4: float
    C3C3: float
    C2C2: float
    C1C1: float
    O4O4: float
    O3O3: float
    NN: float

DistMap = OrderedDict[Tuple[int,int], Dist]
