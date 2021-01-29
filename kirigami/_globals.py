from typing import Dict
import torch

BASES = 'aucg'
N_BASES = len(BASES)
DEFAULT = -1 * torch.ones(N_BASES)
BASE_DICT = defaultdict(lambda: DEFAULT, zip(BASES, torch.eye(N_BASES)))
PairMap = Dict[int,int]
NO_CONTACT = -1

__all__ = ['BASES', 'N_BASES', 'BASE_DICT', 'PairMap', 'NO_CONTACT']
