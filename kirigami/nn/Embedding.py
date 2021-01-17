import re
import torch
from torch import nn
from ..utils.constants import *


class SequenceEmbedding(nn.Module):
    def __init__(self):
        super(SequenceEmbedding, self).__init__()

    def forward(self, sequence):
        one_hot = torch.stack([BASE_DICT[char] for char in sequence.lower()])
        ret = torch.empty(2*N_BASES, len(one_hot), len(one_hot))
        for i in range(len(one_hot)):
            for j in range(len(one_hot)):
                ret[:,i,j] = torch.cat((one_hot[i], one_hot[j]))
        ret = ret.unsqueeze(0)
        return ret


class LabelEmbedding(nn.Module):
    def __init__(self):
        super(LabelEmbedding, self).__init__()

    def forward(self, label):
        lines = label.splitlines()
        matches = re.findall(r'[\d]+$', lines[0])
        L = int(matches[0])
        ret = torch.zeros(L, L)
        for line in lines:
            if line.startswith('#') or line.startswith('i'):
                continue
            line_split = line.split()
            idx1, idx2 = int(line_split[0]), int(line_split[-1])
            ret[idx1-1, idx2-1] = 1.
        ret = ret.unsqueeze(0)
        return ret


class BpseqEmbedding(nn.Module):
    def __init__(self):
        super(BpseqEmbedding, self).__init__()
        self.seq_embed = SequenceEmbedding()

    def forward(self, bpseq):
        lines = bpseq.splitlines()
        L = len(lines)
        idx_ret = torch.zeros(L, L)
        seq = ''
        for line in lines:
            if line.startswith('#'):
                continue
            i, base, j = line.split()
            idx_ret[int(i)-1, int(j)-1] = 1.
            seq += base
        seq_ret = self.seq_embed(seq)
        idx_ret = idx_ret.unsqueeze(0).unsqueeze(0)
        return seq_ret, idx_ret
