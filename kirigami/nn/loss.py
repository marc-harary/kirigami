from typing import *
from numbers import Real
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from kirigami.nn.utils import *
from torch.nn.functional import tanh


# __all__ = ["InverseLoss", "LossEmbedding", "WeightLoss", "ForkL1", "ForkLoss", "CEMulti"]
__all__ = ["InverseLoss", "LossEmbedding", "WeightLoss", "ForkLoss", "CEMulti"]


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
        # self.weight = nn.Parameter(torch.rand(1).cuda())

    def forward(self,
                prd: torch.Tensor,
                grd: torch.Tensor) -> torch.Tensor:
        loss = self._loss(prd, grd)
        loss[grd == 0] *= self._weight
        loss[grd == 1] *= 1 - self._weight
        # return loss.sum()
        return loss.mean()


class CEMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, input, target):
        target_idxs = target.argmax(1)
        loss = F.nll_loss(F.log_softmax(input, dim=1), target_idxs, reduction="none")
        return loss[~target[:,0,...].isnan()].sum()


class ForkLoss(nn.Module):
    def __init__(self,
                 dist_crit: nn.Module,
                 pos_weight: float = None,
                 bin_weight: float = None,
                 inv_weight: float = None,
                 dropout: bool = False,
                 f = None):
        super().__init__()
        # assert 0.0 <= dist_weight <= 1.0
        self._dist_crit = dist_crit
        self.pos_weight = pos_weight
        self.bin_weight = bin_weight
        self.inv_weight = inv_weight


    def forward(self, prd: tuple, grd: tuple, dist_weight: Optional[float] = None):
        prd_con, prd_bin, prd_inv = prd
        grd_con, grd_bin, grd_inv = grd

        con_loss_tens = F.binary_cross_entropy(prd_con, grd_con, reduction="none")
        con_loss_tens[grd_con == 0] *= self.pos_weight
        con_loss_tens[grd_con == 1] *= 1 - self.pos_weight
        # con_loss = con_loss_tens.sum()
        con_loss = con_loss_tens.mean()

        tot_loss = (1-self.inv_weight-self.bin_weight) * con_loss
        bin_loss_all = 0
        inv_loss_all = 0

        for i, (grd_dist, prd_dist) in enumerate(zip(grd_bin, prd_bin)):
            prd_dist[grd_dist.isnan()] = 0
            grd_dist[grd_dist.isnan()] = 0
            dist_loss = F.cross_entropy(prd_dist, grd_dist.argmax(1))#, reduction="sum")
            bin_loss_all += dist_loss.item()
            tot_loss += self.bin_weight * dist_loss

        for i, (grd_dist, prd_dist) in enumerate(zip(grd_inv, prd_inv)):
            diff = prd_dist - grd_dist
            dist_loss = diff + F.softplus(-2.*diff) - log(2.)
            dist_loss = torch.abs(diff)
            # dist_loss = dist_loss[~grd_dist.isnan()].sum()
            dist_loss = dist_loss[~grd_dist.isnan()].mean()
            inv_loss_all += dist_loss.item()
            tot_loss += self.inv_weight * dist_loss

        return tot_loss, bin_loss_all, inv_loss_all, con_loss.item()
