from typing import Dict
from collections import defaultdict
import torch

BASES = 'AUCG'
N_BASES = len(BASES)
DEFAULT = -1 * torch.ones(N_BASES)
BASE_DICT = defaultdict(lambda: DEFAULT, zip(BASES, torch.eye(N_BASES)))
NO_CONTACT = -1
PairMap = Dict[int,int]

__all__ = ['BASES', 'N_BASES', 'BASE_DICT', 'PairMap', 'NO_CONTACT']
