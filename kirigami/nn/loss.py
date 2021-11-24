from typing import *
from numbers import Real
import torch
import torch.nn as nn
from kirigami.nn.utils import *


__all__ = ["InverseLoss", "LossEmbedding", "ForkLoss"]


class InverseLoss(nn.Module):
    def __init__(self, A: float =  20., epsilon: float = 1e-3): # or 1e-4
        super().__init__()
        self.A = A
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, grnd: torch.Tensor) -> torch.Tensor:
        # which loss function to use now?
        """
        6/4/21:
        have network directly output `pred_inv` and have y_hat be `grnd_inv`
        """
        pred_inv = self.A / (pred + self.epsilon)
        grnd_inv = self.A / (grnd + self.epsilon)
        return ((pred_inv - grnd_inv)**2).sum()


class LossEmbedding(nn.Module):
    def __init__(self,
                 max_dist: float = 22.,
                 min_dist: float = 8.,
                 step_dist: float = 0.1):
        super().__init__()
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.step_dist = step_dist 
        self.embedding_dict = torch.eye((self.max_dist - self.min_dist) / self.step_dist)
    
    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(ipt)
        out.unsqueeze_(-1)
        out = out.expand(-1)
        out = out.expand(self.embedding_dict.shape[0])
        for i in range(ipt.shape(-2)):
            for j in range(ipt.shape(-1)):
                dist = ipt[:, i, j]
                idx = (dist - self.min_dist) / self.step_dist
                ipt[:, i, j] = self.embedding_dict[idx]


class ForkLoss(AtomicModule):
    def __init__(self,
                 contact_module: nn.Module,
                 inv_module: nn.Module,
                 one_hot_module: nn.Module,
                 contact_weight: Real,
                 inv_weight: Real,
                 one_hot_weight: Real) -> None:
        assert contact_weight + inv_weight + one_hot_weight == 1
        super().__init__()
        self.contact_module = contact_module
        self.inv_module = inv_module
        self.one_hot_module = one_hot_module
        self.contact_weight = contact_weight
        self.inv_weight = inv_weight
        self.one_hot_weight = one_hot_weight

    def forward(self, pred: Triple[torch.Tensor], grnd: Triple[torch.Tensor]) -> Real:
        out = 0.
        out += self.contact_module(pred[0], grnd[0])
        out += self.inv_module(pred[1], grnd[1])
        out += self.one_hot_module(pred[2], grnd[2])
        return out
