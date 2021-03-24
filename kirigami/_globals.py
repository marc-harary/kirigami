from typing import Dict
from collections import defaultdict, namedtuple
import torch

__all__ = ["BASES",
           "N_BASES",
           "BASE_DICT",
           "INV_BASE_DICT",
           "DEVICE",
           "NO_CONTACT",
           "PairMap",
           "Scores"]

BASES = "AUCG"
N_BASES = len(BASES)
NO_CONTACT = -1
DEFAULT = -1 * torch.ones(N_BASES)
BASE_DICT = defaultdict(lambda: DEFAULT, zip(BASES, torch.eye(N_BASES)))
INV_BASE_DICT = defaultdict(lambda: DEFAULT, zip(torch.eye(N_BASES), BASES))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PairMap = Dict[int,int]
Scores = namedtuple("Scores", ["tp","tn","fp","fn","f1","mcc","ground_pairs","pred_pairs"])
