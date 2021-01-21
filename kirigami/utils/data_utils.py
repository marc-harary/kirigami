import torch
from torch.utils.data import Dataset, DataLoader
from nn.Embedding import *

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
    def __init__(self, list_file: str, embed: AbstractEmbedding):
        super(AbstractDataset, self).__init__()
        self.embed = embed
        with open(list_file, 'r') as f:
            self.files = f.read().splitlines()
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int):
        file = self.files[idx]
        with open(file, 'r') as f:
            file_str = f.read()
        return self.embed(file_str)

class FastaDataset(AbstractASCIIDataset):
    def __init__(self, list_file: str):
        super(FastaDataset, self).__init__(list_file, SequenceEmbedding())

class LabelDataset(AbstractASCIIDataset):
    def __init__(self, list_file: str):
        super(FastaDataset, self).__init__(list_file, LabelEmbedding())

class BpseqDataset(AbstractASCIIDataset):
    def __init__(self, list_file: str):
        super(BpseqDataset, self).__init__(list_file, BpseqEmbedding())
