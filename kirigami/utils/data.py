from pathlib import Path
from typing import Callable, Tuple, Union
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from kirigami.utils.convert import sequence2tensor, label2tensor, bpseq2tensor, st2tensor
from kirigami._globals import DEVICE


__all__ = ["EmbeddedDataset",
           "AbstractASCIIDataset",
           "FastaDataset",
           "LabelDataset",
           "BpseqDataset"]


class EmbeddedDataset(Dataset):
    """Stores pre-embedded files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device = DEVICE,
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
                 batch_load: bool = True,
                 quiet: bool = False):
        super().__init__()
        with open(list_file, "r") as f:
            self.files = f.read().splitlines()
        self.batch_load = batch_load
        if self.batch_load:
            loop = files
            if not quiet:
                loop = tqdm(files)
                print("Embedding files...")
            for file in loop:
                self.data.append(self._load(file))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_load:
            return self.data[idx]
    
    def _load(self, file: str) -> Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]:  
        with open(file, "r") as f:
                txt = f.read()
        emb = embedding(txt)
        if isinstance(emb, tuple):
            return tuple(map(lambda x: x.to(device), emb))
        return emb.to(device) 
        

class FastaDataset(AbstractASCIIDataset):
    """loads and embeds `FASTA` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False) -> None:
        super(FastaDataset, self).__init__(list_file, sequence2tensor, device, batch_load, quiet)


class LabelDataset(AbstractASCIIDataset):
    """loads and embeds `label` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 batch_load: bool = True,
                 quiet: bool = False) -> None:
        super().__init__(list_file, label2tensor, device, batch_load, quiet)


class BpseqDataset(AbstractASCIIDataset):
    """loads and embeds `bpseq` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 batch_load: bool = True,
                 quiet: bool = False) -> None:
        super().__init__(list_file, bpseq2tensor, device, batch_load, quiet)


class StDataset(AbstractASCIIDataset):
    """loads and embeds `st` files"""
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 batch_load: bool = True,
                 quiet: bool = False) -> None:
        super().__init__(list_file, st2tensor, device, batch_load, quiet)
