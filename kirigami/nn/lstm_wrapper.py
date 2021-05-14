import torch
import torch.nn as nn


__all__ = ["LSTM_Wrapper"]


class LSTM_Wrapper(nn.Module):
    """very lightly wraps `torch.nn.LSTM` module for ease of use in `Sequential` objects"""
    
    lstm: nn.LSTM

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(**kwargs)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        print(ipt.shape)
        if ipt.dim() > 3:
            ipt.squeeze_(1)
        out, _ = self.lstm(ipt)
        return out
