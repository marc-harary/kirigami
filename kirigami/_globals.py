from collections import defaultdict, namedtuple
import torch


__all__ = ["BASE_CHARS",
           "N_BASES",
           "BASE_TENSORS",
           "BASE_TUPLES",

           "NO_CONTACT",
           "DEFAULT",

           "CHAR2TENSOR",
           "CHAR2TUPLE",
           "CHAR2IDX",

           "TENSOR2CHAR",
           "TUPLE2CHAR",
           "IDX2CHAR",

           "CANONICAL_CHARS",
           "CANONICAL_TUPLES"]


BASE_CHARS = "AUCG"
N_BASES = len(BASE_CHARS)
BASE_TENSORS = [i for i in torch.eye(N_BASES)]
BASE_TUPLES = [tuple(i.tolist()) for i in torch.eye(N_BASES)]

NO_CONTACT = -1
DEFAULT = -1 * torch.ones(N_BASES)

CHAR2TENSOR = defaultdict(lambda: DEFAULT, zip(BASE_CHARS, BASE_TENSORS))
CHAR2TUPLE = defaultdict(lambda: DEFAULT, zip(BASE_CHARS, BASE_TUPLES))
CHAR2IDX = {"A": 0, "U": 1, "C": 2, "G": 3}

TENSOR2CHAR = defaultdict(lambda: DEFAULT, zip(BASE_TENSORS, BASE_CHARS))
TUPLE2CHAR = defaultdict(lambda: DEFAULT, zip(BASE_TUPLES, BASE_CHARS))
IDX2CHAR = {0: "A", 1: "U", 2: "C", 3: "G"}

CANONICAL_CHARS = ["AU", "UA", "CG", "GC", "UG", "GU"]
CANONICAL_TUPLES = [(BASE_TUPLES[0], BASE_TUPLES[1]),
                    (BASE_TUPLES[1], BASE_TUPLES[0]),
                    (BASE_TUPLES[2], BASE_TUPLES[3]),
                    (BASE_TUPLES[3], BASE_TUPLES[2]),
                    (BASE_TUPLES[1], BASE_TUPLES[3]),
                    (BASE_TUPLES[3], BASE_TUPLES[1])]
