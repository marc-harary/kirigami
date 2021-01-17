import torch
from torch import nn

class Maximizer(nn.Module):
    '''Sets max value in row to 1 and all others to 0'''
    def __init__(self, thres=.5):
        super(Maximizer, self).__init__()
        self.thres = thres

    def forward(self, input):
        L = input.shape[-1]
        input.diagonal(0, 2, 3)[:] = float('-inf')
        out = torch.eye(L).reshape_as(input)
        max_obj = torch.max(input, 3)
        mask = max_obj.values > self.thres
        idx1 = torch.arange(0, L)[mask.squeeze()]
        idx2 = max_obj.indices[mask]
        out[:,:,idx1,idx2] = 1.
        out[:,:,idx2,idx1] = 1.
        return out
