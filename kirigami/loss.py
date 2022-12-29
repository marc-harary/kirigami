from typing import *
from numbers import Real
import numpy as np
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kirigami.nn.utils import *
from torch.nn.functional import tanh


__all__ = ["ForkLoss"]


class ForkLoss(nn.Module):
    def __init__(self,
                 pos_weight: float,
                 con_weight: float = 1.,
                 dists = None,
                 use_logit = False):
        super().__init__()
        assert 0.0 <= con_weight <= 1.0

        self.pos_weight = pos_weight
        self.con_weight = con_weight
        self.dists = [] if dists is None else dists
        self.use_logit = use_logit

    def forward(self, prd: dict, grd: dict):
        loss_dict = {}

        # contact loss
        grd_con_ = grd["con"].clone().to(prd["con"].device)
        grd_con_ = grd_con_.reshape_as(prd["con"])
        mask = ~grd_con_.isnan()
        grd_con_[~mask] = 0. # zero out nan's so BCE loss doesn't throw error
        if self.use_logit:
            con_loss_tens = F.binary_cross_entropy_with_logits(prd["con"], grd_con_, reduction="none")
        else:
            con_loss_tens = F.binary_cross_entropy(prd["con"], grd_con_, reduction="none")
        con_loss_tens[grd_con_ == 1] *= self.pos_weight
        con_loss_tens[grd_con_ == 0] *= 1 - self.pos_weight
        loss_dict["con"] = self.con_weight * con_loss_tens[mask].mean()
        tot_loss = self.con_weight * con_loss_tens[mask].mean()

        loss_dict["dists"] = {}
        for ((key, prd_val), (_, grd_val)) in zip(prd["dists"].items(), grd["dists"].items()):
            grd_val = grd_val["bin"]
            mask = ~grd_val.isnan()[:, 0, ...]
            loss_tens = F.cross_entropy(prd_val, grd_val.argmax(1), reduction="none")
            loss_dict["dists"][key] = loss_tens[mask].mean()
            tot_loss += (1 - self.con_weight) * loss_dict["dists"][key]

        loss_dict["tot"] = tot_loss 

        return loss_dict

