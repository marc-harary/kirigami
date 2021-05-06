from pathlib import Path
from typing import Callable, Tuple, Union
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from kirigami.utils.convert import *
from kirigami._globals import *


__all__ = ["EmbeddedDataset",
           "AbstractASCIIDataset",
           "DenseDataset",
           "SparseDataset",
           "BpseqDataset",
           "StDataset"]


class DenseDataset(Dataset[torch.Tensor]):
    """stores contracted sequences"""

    seqs: Tuple[torch.Tensor]
    labs: Tuple[torch.Tensor]

    def __init__(self, seqs: torch.Tensor, labs: torch.Tensor) -> None:
        super().__init__()
        assert(len(seqs) == len(labs))
        self.seqs = seqs
        self.labs = labs
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # seq = dense2volume(self.seqs[idx])
        # lab = self.labs[idx]
        # return seq, lab
        return self.seqs[idx], self.labs[idx]

    def __len__(self) -> int:
        return self.seqs.shape[0]


class SparseDataset(Dataset[torch.Tensor]):
    """stores sparse conracted sequences"""

    seqs: Tuple[torch.Tensor]
    labs: Tuple[torch.Tensor]
    seq_densify: Callable
    lab_densify: Callable

    def __init__(self, seqs: torch.Tensor, labs: torch.Tensor) -> None:
        super().__init__()
        assert(len(seqs) == len(labs))
        self.seqs = seqs
        self.labs = labs
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # seq = dense2volume(self.seqs[idx].to_dense())
        return self.seqs[idx].to_dense(), self.labs[idx].to_dense()

    def __len__(self) -> int:
        return self.seqs.shape[0]


class EmbeddedDataset(Dataset):
    """stores pre-embedded files stored on disk"""

    list_file: Path
    device: torch.device
    quiet: bool
    max_len: int

    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False,
                 max_len: int = 0,
                 batch_load: bool = True) -> None:
        super().__init__() 
        self.device = device
        self.batch_load = batch_load
        with open(list_file, "r") as f:
            self.files = f.read().splitlines()
        if self.batch_load:
            loop = tqdm(self.files) if not quiet else self.files
            self.data = []
            for file in loop:
                self.data.append(torch.load(file)) 
    
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_load:
            return self.data[idx]
        seq, lab = torch.load(self.files[idx])
        # return seq.to(self.device), lab.to(self.device)
        return seq, lab
        

class AbstractASCIIDataset(Dataset):
    """abstract class for all ASCII-encoded datasets"""

    list_file: Path
    embedding: Callable
    device: torch.device
    max_len: int
    batch_load: bool

    def __init__(self,
                 list_file: Path,
                 embedding: Callable,
                 device: torch.device,
                 max_len: int = 0,
                 batch_load: bool = True):
        super().__init__()
        self.embedding = embedding
        self.device = device
        self.max_len = max_len
        self.batch_load = batch_load
        with open(list_file, "r") as f:
            self.files = f.read().splitlines()
        if self.batch_load:
            embeds = [self._load(file) for file in self.files]
            seqs = [seq for seq, _ in embeds]
            labs = [lab for _, lab in embeds]
            self.seqs = [sequence2tensor(seq, max_len=self.max_len, device=self.device) for seq in seqs]
            self.labs = [pairmap2tensor(lab, max_len=self.max_len, device=self.device) for lab in labs]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_load:
            return self.seqs[idx], self.labs[idx]
        seq, lab = self._load(self.files[idx]) 
        seq_emb = sequence2tensor(seq, max_len=self.max_len, device=self.device)
        lab_emb = pairmap2tensor(lab, max_len=self.max_len, device=self.device)
        return seq_emb, lab_emb

    def _load(self, file: str) -> Tuple[str, PairMap]:
        with open(file, "r") as f:
            txt = f.read()
        return self.embedding(txt)
        

# class FastaDataset(AbstractASCIIDataset):
#     """loads and embeds `FASTA` files"""
#     def __init__(self,
#                  list_file: Path,
#                  device: torch.device):
#         super(FastaDataset, self).__init__(list_file, sequence2tensor, device, batch_load)
# 
# 
# class LabelDataset(AbstractASCIIDataset):
#     """loads and embeds `label` files"""
#     def __init__(self,
#                  list_file: Path,
#                  device: torch.device,
#                  batch_load: bool = True):
#         super().__init__(list_file, label2tensor, device, batch_load)


class BpseqDataset(AbstractASCIIDataset):
    """loads and embeds `bpseq` files"""

    list_file: Path
    device: torch.device
    max_len: bool
    batch_load: bool

    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 max_len: bool = False,
                 batch_load: bool = True):
        super().__init__(list_file, bpseq2pairmap, device, max_len, batch_load)


class StDataset(AbstractASCIIDataset):
    """loads and embeds `st` files"""

    list_file: Path
    device: torch.device
    max_len: bool = False
    batch_load: bool = True

    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 max_len: bool = False,
                 batch_load: bool = True):
        super().__init__(list_file, st2pairmap, device, max_len, batch_load)
