from typing import *
from abc import *
import re
from multipledispatch import dispatch
import torch
from torch import nn
from ..utils.constants import *

class AbstractEmbedding(ABC, nn.Module):
    @dispatch(str)
    @abstractmethod
    def forward(self, data_str: str) -> torch.Tensor:
        pass
    
    @dispatch(list)
    @abstractmethod
    def forward(self, data_list: List[str]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], List[int]]:
        pass
   
# singleton should be of shape (8, L, L)
class SequenceEmbedding(AbstractEmbedding):
    @dispatch(str)
    def forward(self, sequence):
        one_hot = torch.stack([BASE_DICT[char] for char in sequence.lower()])
        ret = torch.empty(2*N_BASES, len(one_hot), len(one_hot)) 
        for i in range(len(one_hot)):
            for j in range(len(one_hot)):
                ret[:,i,j] = torch.cat((one_hot[i], one_hot[j]))
        return ret.unsqueeze(0)

    @dispatch(list)
    def forward(self, sequence_list):
        return [self.forward(sequence) for sequence in sequence_list]
    

# class SequenceEmbedding(AbstractEmbedding):
#     '''Embeds either single or multiple FASTA sequences as one-hot tensors'''
#     @dispatch(str)
#     def forward(self, sequence): 
#         return torch.stack([BASE_DICT[char] for char in sequence.lower()])
#     
#     @dispatch(list)
#     def forward(self, sequences) -> Tuple[torch.Tensor, List[int]]:
#         lengths = [len(seq) for seq in sequences]
#         tensors = [self.forward(seq) for seq in sequences]
#         embed = nn.utils.rnn.pad_sequence(tensors, batch_first=True)
#         embed = embed.permute(0, 2, 1)
#         return embed, lengths

class LabelEmbedding(AbstractEmbedding):
    '''Embeds either single or multiple `bpseq` files as one-hot tensors'''
    @dispatch(str)
    def forward(self, label):
        lines = label.splitlines()
        matches = re.findall(r'[\d]+$', lines[0])
        L = int(matches[0])
        ret = torch.zeros(L, L).to(torch.int64)
        for line in lines:
            if line.startswith('#') or line.startswith('i'):
                continue
            idx1, idx2 = [int(item) for item in line.split()]
            ret[idx1-1, idx2-1] = 1
        return ret
    
    @dispatch(list)
    def forward(self, labels) -> Tuple[List[torch.Tensor], List[int]]:
        embed = [self.forward(label) for label in labels]
        lengths = [len(mat) for mat in embed]
        return embed, lengths
