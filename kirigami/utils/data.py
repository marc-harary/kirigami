from pathlib import Path
from typing import Callable, Tuple, Union
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from kirigami.utils.convert import sequence2tensor, label2tensor, bpseq2tensor, st2tensor


__all__ = ["EmbeddedDataset",
           "AbstractASCIIDataset",
           "FastaDataset",
           "LabelDataset",
           "BpseqDataset",
           "StDataset"]


class EmbeddedDataset(Dataset):
    """Stores pre-embedded files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False,
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
        return seq.to(self.device), lab.to(self.device)
        

class AbstractASCIIDataset(Dataset):
    """abstract class for all ASCII-encoding datasets"""
    def __init__(self,
                 list_file: Path,
                 embedding: Callable,
                 device: torch.device,
                 batch_load: bool = True):
        super().__init__()
        self.embedding = embedding
        self.device = device
        self.batch_load = batch_load
        with open(list_file, "r") as f:
            self.files = f.read().splitlines()
        if self.batch_load:
            self.data = [self._load(file) for file in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_load:
            return self.data[idx]
        return self._load(self.files[idx]) 

    def to(self, device: torch.device) -> None:
        self.device = device
        if self.batch_load:
            if isinstance(self.data[0], tuple):
                self.data = [(tup[0].to(device),tup[1].to(device)) for tup in self.data]
            else:
                self.data = [tens.to(device) for tens in self.data]
    
    def _load(self, file: str) -> Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]:  
        with open(file, "r") as f:
            txt = f.read()
        emb = self.embedding(txt)
        if isinstance(emb, tuple):
            return tuple(map(lambda x: x.to(self.device), emb))
        return emb.to(self.device) 
        

class FastaDataset(AbstractASCIIDataset):
    """loads and embeds `FASTA` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device):
        super(FastaDataset, self).__init__(list_file, sequence2tensor, device, batch_load)


class LabelDataset(AbstractASCIIDataset):
    """loads and embeds `label` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 batch_load: bool = True):
        super().__init__(list_file, label2tensor, device, batch_load)


class BpseqDataset(AbstractASCIIDataset):
    """loads and embeds `bpseq` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 batch_load: bool = True):
        super().__init__(list_file, bpseq2tensor, device, batch_load)


class StDataset(AbstractASCIIDataset):
    """loads and embeds `st` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 batch_load: bool = True):
        super().__init__(list_file, st2tensor, device, batch_load)
