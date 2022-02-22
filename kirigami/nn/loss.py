from typing import *
from numbers import Real
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from kirigami.nn.utils import *
from torch.nn.functional import tanh


__all__ = ["InverseLoss", "LossEmbedding", "WeightLoss", "ForkL1"]


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


class WeightLoss(nn.Module):
    """Implements weighted binary cross-entropy loss"""
    def __init__(self, weight: float) -> None:
        super().__init__()
        assert 0.0 <= weight <= 1.0
        self._weight = weight
        self._loss = nn.BCELoss(reduction="none")

    def forward(self,
                prd: torch.Tensor,
                grd: torch.Tensor) -> torch.Tensor:
        loss = self._loss(prd, grd)
        loss[grd == 1] *= self._weight
        loss[grd == 0] *= (1 - self._weight)
        return loss.sum()


# class ForkLoss(nn.Module):
#     def __init__(self,
#                  pos_weight: float,
#                  dist_weight: float,
#                  n_dists: int = 10) -> None:
#         super().__init__()
#         assert 0.0 <= dist_weight <= 1.0
#         self._weight_loss = WeightLoss(pos_weight)
#         self._dist_weight = dist_weight
#         self._n_dists = n_dists 
#         self._dist_loss = nn.L1Loss(reduction="none")
#         # self._dist_loss = nn.MSELoss(reduction="none")
#         
#     def forward(self,
#                 prd: Tuple[torch.Tensor, torch.Tensor],
#                 grd: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         prd_con, prd_dist = prd
#         grd_con, grd_dist = grd
#         con_loss = self._weight_loss(prd_con, grd_con) 
#         length = prd_dist.shape[-1]
#         diags = torch.arange(length)
#         # diffs = self._dist_loss(1/prd_dist, 1/grd_dist)
#         diffs = self._dist_loss(prd_dist, grd_dist)
#         diffs[:, :, diags, diags] = 0
#         # diffs[grd_dist < 0] = 0
#         diffs[grd_dist == 0] = 0
#         diffs *= torch.exp(-3*grd_dist)
#         # L1 loss
#         dist_loss = torch.sum(diffs)
#         return (self._dist_weight*dist_loss + (1-self._dist_weight)*con_loss,
#                 dist_loss.item(),
#                 con_loss.item())

# class ForkLogCosh(nn.Module):
#     def __init__(self, pos_weight: float, dist_weight: float):
#         super().__init__()
#         assert 0.0 <= dist_weight <= 1.0
#         self._log2 = log(2)
#         self._weight_loss = WeightLoss(pos_weight)
#         self._dist_weight = dist_weight
#         self._dist_loss = nn.L1Loss(reduction="none")
# 
# 
#     def forward(self, prd: tuple, grd: tuple):
#         prd_con, prd_dist = prd
#         grd_con, grd_dist = grd
# 
#         # prd_dist = 1 / prd_dist 
#         # grd_dist = 1 / grd_dist 
#         # diff = prd_dist - grd_dist
#         # diff *= self._K
#         # dist_loss = diff + F.softplus(-2.*diff) - self._log2
#         # dist_loss = torch.abs(diff)
#         con_loss = self._weight_loss(prd_con, grd_con) 
# 
#         dist_loss = self._dist_loss(prd_dist, grd_dist)
#         length = prd_dist.shape[-1]
#         diags = torch.arange(length)
#         dist_loss[:, :, diags, diags] = 0.
#         dist_loss[grd_dist <= 0] = 0.
#         dist_loss[grd_dist == 1] = 0.
#         dist_loss = torch.sum(dist_loss)
# 
#         tot_loss = self._dist_weight*dist_loss + (1.-self._dist_weight)*con_loss
# 
#         return tot_loss, dist_loss.item(), con_loss.item()


class ForkL1(nn.Module):
    def __init__(self,
                 pos_weight: float,
                 dist_weight: float,
                 dropout: bool = True,
                 inv: bool = False,
                 f = None):
        super().__init__()
        assert 0.0 <= dist_weight <= 1.0
        self._dropout = dropout
        self._dist_weight = dist_weight
        self._weight_crit = WeightLoss(pos_weight)
        self._dist_crit = nn.L1Loss(reduction="none")
        self._inv = inv
        self._f = f if f else lambda x: x


    def forward(self, prd: tuple, grd: tuple):
        prd_con, prd_dist = prd
        grd_con, grd_dist = grd

        if self._inv:
            grd_dist_ = 1 / (grd_dist + 1e-16)
            grd_dist_[grd_dist == 0] = 0 
            prd_dist_ = 1 / (prd_dist + 1e-16)
            dist_loss = self._dist_crit(prd_dist_, grd_dist_)
        else:
            dist_loss = self._dist_crit(prd_dist, grd_dist)
        con_loss = self._weight_crit(prd_con, grd_con) 

        length = prd_dist.shape[-1]
        diags = torch.arange(length)
        dist_loss[:, :, diags, diags] = 0.
        dist_loss[grd_dist <= 0] = 0.
        if self._dropout:
            probs = torch.rand(grd_dist.shape, device=grd_dist.device)
            mask = probs < self._f(grd_dist)
            dist_loss[mask] = 0.
        dist_loss[grd_dist == 1] = 0.
        dist_loss = torch.sum(dist_loss)

        tot_loss = self._dist_weight*dist_loss + (1.-self._dist_weight)*con_loss

        return tot_loss, dist_loss.item(), con_loss.item()
