from typing import List
import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self, func_type: str, params: dict):
        loss_class = getattr(nn, func_type)
        self.loss_func = loss_class(**params)

    def forward(self, predict: torch.Tensor, lengths: List[int], ground: torch.Tensor):
        tensors_zeroed = []
        for tensor, length in zip(predict, lengths):
            tensor[:, length:] = 0.
            tensors_zeroed.append(tensor)
        return self.loss_func(tensors_zeroed, ground)
