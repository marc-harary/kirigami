from typing import Tuple
import torch
from kirigami.utils.utilities import *


__all__ = ['AbstractEmbedding', 'LabelEmbedding', 'BpseqEmbedding', 'SequenceEmbedding']


class AbstractEmbedding(torch.nn.Module):
    def __init__(self):
        super(AbstractEmbedding, self).__init__()


class SequenceEmbedding(AbstractEmbedding):
    '''Embeds FASTA file'''
    def __init__(self):
        super(SequenceEmbedding, self).__init__()

    def forward(self, sequence: str) -> torch.Tensor:
        return seq2tensor(sequence)


class LabelEmbedding(AbstractEmbedding):
    '''Embeds label files'''
    def __init__(self):
        super(LabelEmbedding, self).__init__()

    def forward(self, label: str) -> torch.Tensor:
        return lab2tensor(label)


class BpseqEmbedding(AbstractEmbedding):
    '''Embeds .bpseq file'''
    def __init__(self):
        super(BpseqEmbedding, self).__init__()
        self.seq_embed = SequenceEmbedding()

    def forward(self, bpseq: str) -> Tuple[str, torch.Tensor]:
        sequence, pair_map =  bpseq2pairs(bpseq)
        pair_tensor = pairs2tensor(pair_map)
        return sequence, pair_tensor

