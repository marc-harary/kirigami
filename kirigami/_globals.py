from typing import OrderedDict
from collections import defaultdict, namedtuple
import torch

__all__ = ["PairMap",
           "Scores",
           "NO_CONTACT",
           "DEFAULT",
           "BASE_CHARS",
           "BASE_TENSORS",
           "N_BASES",
           "CHAR2TENSOR",
           "TENSOR2CHAR",
           "TUPLE2CHAR",
           "CHAR2TUPLE",
           "CANONICAL_CHARS",
           "CANONICAL_TUPLES"]

PairMap = OrderedDict[int,int]
Scores = namedtuple("Scores", ["tp","tn","fp","fn","f1","mcc","ground_pairs","pred_pairs"])
BASE_CHARS = "AUCG"
N_BASES = len(BASE_CHARS)
BASE_TENSORS = [i for i in torch.eye(N_BASES)]
BASE_TUPLES = [tuple(i.tolist()) for i in torch.eye(N_BASES)]
NO_CONTACT = -1
DEFAULT = -1 * torch.ones(N_BASES)
CHAR2TENSOR = defaultdict(lambda: DEFAULT, zip(BASE_CHARS, BASE_TENSORS))
TENSOR2CHAR = defaultdict(lambda: DEFAULT, zip(BASE_TENSORS, BASE_CHARS))
TUPLE2CHAR = defaultdict(lambda: DEFAULT, zip(BASE_TUPLES, BASE_CHARS))
CHAR2TUPLE = defaultdict(lambda: DEFAULT, zip(BASE_CHARS, BASE_TUPLES))
CANONICAL_CHARS = ["AU", "UA", "CG", "GC", "UG", "GU"]
CANONICAL_TUPLES = [(BASE_TUPLES[0], BASE_TUPLES[1]),
                    (BASE_TUPLES[1], BASE_TUPLES[0]),
                    (BASE_TUPLES[2], BASE_TUPLES[3]),
                    (BASE_TUPLES[3], BASE_TUPLES[2]),
                    (BASE_TUPLES[1], BASE_TUPLES[3]),
                    (BASE_TUPLES[3], BASE_TUPLES[1])]
