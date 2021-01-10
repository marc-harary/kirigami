import torch
from torch import nn
from torch.nn import functional as F

class SequenceEmbedding(nn.Module):
    def _forward(self, sequence):
        return torch.stack([BASE_DICT[char] for char in sequence.lower()])
    
    def forward(self, sequences, max_length_dataset=0):
        lengths = list(map(len, sequences))
        fastas = list(map(self._forward, sequences))
        embed = nn.utils.rnn.pad_sequence(fastas)
        embed = embed.permute(1, 2, 0)
        diff = max(max_length_dataset - max(lengths), 0)
        embed_pad = F.pad(embed, (0, diff), 'constant', 0)
        return embed_pad, lengths