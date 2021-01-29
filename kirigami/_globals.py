from collections import defaultdict
from typing import List, Tuple
import torch

BASES = 'aucg'
N_BASES = len(BASES)
DEFAULT = -1 * torch.ones(N_BASES)
BASE_DICT = defaultdict(lambda: DEFAULT, zip(BASES, torch.eye(N_BASES)))

__all__ = ['BASES', 'N_BASES', 'BASE_DICT']
