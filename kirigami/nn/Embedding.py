import re
import torch
from torch import nn
from utils.constants import *


class AbstractEmbedding(nn.Module):
    def __init__(self):
        super(AbstractEmbedding, self).__init__()


class SequenceEmbedding(AbstractEmbedding):
    def __init__(self):
        super(SequenceEmbedding, self).__init__()

    def forward(self, sequence: str) -> torch.Tensor:
        one_hot = torch.stack([BASE_DICT[char] for char in sequence.lower()])
        out = torch.empty(2*N_BASES, len(one_hot), len(one_hot))
        for i in range(len(one_hot)):
            for j in range(len(one_hot)):
                out[:, i, j] = torch.cat((one_hot[i], one_hot[j]))
        return out


class LabelEmbedding(AbstractEmbedding):
    def __init__(self):
        super(LabelEmbedding, self).__init__()

    def forward(self, label: str) -> torch.Tensor:
        lines = label.splitlines()
        matches = re.findall(r'[\d]+$', lines[0])
        L = int(matches[0])
        out = torch.zeros(L, L)
        for line in lines:
            if line.startswith('#') or line.startswith('i'):
                continue
            line_split = line.split()
            idx1, idx2 = int(line_split[0]), int(line_split[-1])
            out[idx1-1, idx2-1] = 1.
        out = out.unsqueeze(0)
        return out


class BpseqEmbedding(AbstractEmbedding):
    def __init__(self):
        super(BpseqEmbedding, self).__init__()
        self.seq_embed = SequenceEmbedding()

    def forward(self, bpseq: str) -> torch.Tensor:
        lines = bpseq.splitlines()
        lines = list(filter(lambda line: not line.startswith('#'), lines))
        L = len(lines)
        idx_out = torch.zeros(L, L)
        seq = ''
        for line in lines:
            i, base, j = line.split()
            idx_out[int(i)-1, int(j)-1] = 1.
            seq += base
        seq_out = self.seq_embed(seq)
        idx_out = idx_out.reshape(1, L, L)
        return seq_out, idx_out
