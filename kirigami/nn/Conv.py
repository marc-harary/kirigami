import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self,
                 dropout_rate,
                 act_func,
                 in_channels,
                 out_channels,
                 kernel_size,
                 **kwargs):
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              **kwargs)
        self.act_func = act_func
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, input):
        ret = input
        ret = self.conv(input)
        ret = self.act_func(ret)
        ret = self.dropout(ret)
        return ret

class LSTMLayer(nn.Module):
   pass 
        
