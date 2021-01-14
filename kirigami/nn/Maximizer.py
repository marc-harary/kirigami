import torch
from torch import nn

class Maximizer(nn.Module):
    '''Sets max value in row to 1 and all others to 0'''
    def forward(self, input):
        ret = input.squeeze()
        max_obj = ret.max(dim=1, keepdim=True)
        ret_filt = (ret == max_obj.values)
        return ret_filt.to(torch.int64)
