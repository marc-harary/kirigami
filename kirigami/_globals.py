from typing import Dict
from collections import defaultdict, namedtuple
import torch

__all__ = ["PairMap",
           "Scores",
           "NO_CONTACT",
           "DEFAULT",
           "BASES_CHAR",
           "BASES_TENSOR",
           "N_BASES",
           "BASES_CHAR_DICT",
           "BASES_TENSOR_DICT",
           "CANONICALS_CHAR",
           "CANONICALS_TENSOR"]

PairMap = Dict[int,int]
Scores = namedtuple("Scores", ["tp","tn","fp","fn","f1","mcc","ground_pairs","pred_pairs"])
BASES_CHAR = "AUCG"
N_BASES = len(BASES_CHAR)
BASES_TENSOR = [tuple(tensor.tolist()) for tensor in torch.eye(N_BASES)]
NO_CONTACT = -1
DEFAULT = -1 * torch.ones(N_BASES)
BASES_CHAR_DICT = defaultdict(lambda: DEFAULT, zip(BASES_CHAR, BASES_TENSOR))
BASES_TENSOR_DICT = defaultdict(lambda: DEFAULT, zip(BASES_TENSOR, BASES_CHAR))
CANONICALS_CHAR = ["AU", "UA", "CG", "GC"]
CANONICALS_TENSOR = [(BASES_TENSOR[0], BASES_TENSOR[1]),
                     (BASES_TENSOR[1], BASES_TENSOR[0]),
                     (BASES_TENSOR[2], BASES_TENSOR[3]),
                     (BASES_TENSOR[3], BASES_TENSOR[2])]
