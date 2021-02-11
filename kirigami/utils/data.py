'''dataset classes for various input files'''


from pathlib import Path
from typing import Callable, Tuple, Union
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from kirigami.utils.convert import sequence2tensor, label2tensor, bpseq2tensor


__all__ = ['EmbeddedDataset',
           'AbstractASCIIDataset',
           'FastaDataset',
           'LabelDataset',
           'BpseqDataset']

class EmbeddedDataset(Dataset):
    '''Stores pre-embedded files'''
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False,
                 batch_load: bool = True) -> None:
        super().__init__() 
        self.pre_load = pre_load
        with open(list_file, 'r') as f:
            self.files = f.read().splitlines()
        if self.batch_load:
            self.data = []
            for file in self.files:
                self.data.append(torch.load(file)) 
    
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_load:
            return self.data[idx]
        return torch.load(self.files[idx])
        

class AbstractASCIIDataset(Dataset):
    '''abstract class for all ASCII-encoding datasets'''
    def __init__(self,
                 list_file: Path,
                 embedding: Callable,
                 device: torch.device,
                 quiet: bool = False,
                 batch_load: bool = True) -> None:
        super().__init__()
        with open(list_file, 'r') as f:
            self.files = f.read().splitlines()
        self.batch_load = batch_load
        if self.batch_load:
            loop = files
            if not quiet:
                loop = tqdm(files)
                print('Embedding files...')
            for file in loop:
                self.data.append(self._load(file))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_load:
            return self.data[idx]
    
    def _load(self, file: str) -> Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]:  
        with open(file, 'r') as f:
                txt = f.read()
        emb = embedding(txt)
        if isinstance(emb, tuple):
            return tuple(map(lambda x: x.to(device), emb))
        return emb.to(device) 
        

class FastaDataset(AbstractASCIIDataset):
    '''loads and embeds `FASTA` files'''
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False) -> None:
        super(FastaDataset, self).__init__(list_file, sequence2tensor, device, quiet)


class LabelDataset(AbstractASCIIDataset):
    '''loads and embeds `label` files'''
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False) -> None:
        super().__init__(list_file, label2tensor, device, quiet)


class BpseqDataset(AbstractASCIIDataset):
    '''loads and embeds `bpseq` files'''
    def __init__(self,
                 list_file: Path,
                 device: torch.device,
                 quiet: bool = False) -> None:
        super().__init__(list_file, bpseq2tensor, device, quiet)
