from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader
from kirigami.utils.utilities import *


__all__ = ['TensorDataset', 'AbstractASCIIDataset']


class TensorDataset(Dataset):
    def __init__(self, list_file: str):
        super(TensorDataset, self).__init__()
        with open(list_file, 'r') as f:
            self.files = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx])


class AbstractASCIIDataset(Dataset):
    def __init__(self, list_file: str, embedding: Callable):
        super(AbstractDataset, self).__init__()
        self.embedding = embedding
        with open(list_file, 'r') as f:
            self.files = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]
        with open(file, 'r') as f:
            file_str = f.read()
        return self.embedding(file_str)


class FastaDataset(AbstractASCIIDataset):
    def __init__(self, list_file: str):
        super(FastaDataset, self).__init__(list_file, seq2tensor)


class LabelDataset(AbstractASCIIDataset):
    def __init__(self, list_file: str):
        super(FastaDataset, self).__init__(list_file, lab2tensor)


class BpseqDataset(AbstractASCIIDataset):
    def __init__(self, list_file: str):
        super(BpseqDataset, self).__init__(list_file, bpseq2tensor)
