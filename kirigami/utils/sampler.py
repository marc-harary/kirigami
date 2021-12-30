from typing import *
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from collections import defaultdict
import random


class LengthSampler(Sampler):
    """ MUST USE WITH BATCH SIZE 1"""
    def __init__(self, data: List):
        super().__init__(data)
        lengths = set([tup[0].shape[1] for tup in data])

        length_dict = {}
        self.batches = {}
        for idx, tup in enumerate(data):
            length = tup[0].shape[1]
            if length not in length_dict:
                length_dict[length] = []
                self.batches[length] = []
            length_dict[length].append(tup)
            self.batches[length].append(idx)

        self.batches = list(self.batches.values())

    def __iter__(self):
        shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class LengthSampler(Sampler):
    def __init__(self, data: List, bin_size: int):
        # super().__init__(lengths)
        super().__init__(data)
        lengths = torch.tensor([tup[0].shape[1] for tup in data])

        ii = lengths.argsort()
        self._set_size = len(data)
        self._bin_size = bin_size
        ncols = int(np.ceil(len(data) / self._bin_size))
        size_round = ncols * self._bin_size
        diff = size_round - len(data)
        idxs = np.full(diff, -1)
        ii = np.hstack((ii, idxs))
        self._ii = ii.reshape((ncols, self._bin_size))

        out_ii = self._ii.copy()
        np.random.shuffle(out_ii)
        for i in range(out_ii.shape[0]):
            np.random.shuffle(out_ii[i,:])
        idxs = out_ii.flatten()
        idxs = idxs[idxs > 0]

        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)


class EqualSampler(Sampler):
    def __init__(self, data: List):
        super().__init__(data)

        length_dict = defaultdict(list)
        for i, (fasta, *_) in enumerate(data):
            length_dict[fasta.shape[1]].append(i)
        self.idxs = list(length_dict.values())
        random.shuffle(self.idxs)

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
