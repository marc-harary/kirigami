from pathlib import Path
from typing import Callable, Tuple
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from kirigami.utils.utilities import *


__all__ = ['AbstractASCIIDataset',
           'FastaDataset',
           'LabelDataset',
           'BpseqDataset']


class AbstractASCIIDataset(Dataset):
    def __init__(self, list_file: Path, embedding: Callable, quiet: bool = False) -> None:
        super(AbstractASCIIDataset, self).__init__()
        with open(list_file, 'r') as f:
            files = f.read().splitlines()
        self.data = []
        print('Embedding files...')
        loop = files if quiet else tqdm(files)
        for file in loop:
            with open(file, 'r') as f:
                txt = f.read()
            self.data.append(embedding(txt))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class FastaDataset(AbstractASCIIDataset):
    def __init__(self, list_file: Path, quiet: bool = False) -> None:
        super(FastaDataset, self).__init__(list_file, sequence2tensor, quiet)


class LabelDataset(AbstractASCIIDataset):
    def __init__(self, list_file: Path, quiet: bool = False) -> None:
        super(FastaDataset, self).__init__(list_file, label2tensor, quiet)


class BpseqDataset(AbstractASCIIDataset):
    def __init__(self, list_file: Path, quiet: bool = False) -> None:
        super(BpseqDataset, self).__init__(list_file, bpseq2tensor, quiet)
