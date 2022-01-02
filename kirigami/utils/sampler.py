from typing import *
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from collections import defaultdict
import random


class EqualSampler(Sampler):
    def __init__(self, data: List):
        super().__init__(data)

        length_dict = defaultdict(list)
        for i, (fasta, *_) in enumerate(data):
            length_dict[fasta.shape[1]].append(i)
        self.idxs = list(length_dict.values())

    def __iter__(self):
        random.shuffle(self.idxs)
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
