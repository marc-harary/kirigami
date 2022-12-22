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


# class ForkLoss(nn.Module):
#     def __init__(self,
#                  # bins: torch.tensor,
#                  pos_weight: float,
#                  con_weight: float,
#                  bin_weight: float,
#                  inv_weight: float,
#                  n_dists: int = 10):
#         super().__init__()
#         # assert 0.0 <= dist_weight <= 1.0
#         assert (con_weight + bin_weight + inv_weight) == 1.
# 
#         self.pos_weight = pos_weight
#         self.con_weight = con_weight
#         self.bin_weight = bin_weight
#         self.inv_weight = inv_weight
#         self.n_dists = n_dists
# 
# 
#     def forward(self, prd, grd):
#         prd_con, prd_bin, prd_inv = prd
#         grd_con, grd_bin, grd_inv = grd
# 
#         # contact loss
#         prd_con[grd_con.isnan()] = 0
#         grd_con[grd_con.isnan()] = 0
#         con_loss_tens = F.binary_cross_entropy(prd_con, grd_con, reduction="none")
#         con_loss_tens[grd_con == 0] *= self.pos_weight
#         con_loss_tens[grd_con == 1] *= 1 - self.pos_weight
#         con_loss = con_loss_tens.mean()
# 
#         # bin loss
#         prd_bin[grd_bin.isnan()] = 0
#         grd_bin[grd_bin.isnan()] = 0
#         bin_loss_tens = F.cross_entropy(prd_bin, grd_bin.argmax(1), reduction="none")
#         bin_loss = bin_loss_tens.mean()
# 
#         # inv loss
#         prd_inv[grd_inv.isnan()] = 0
#         grd_inv[grd_inv.isnan()] = 0
#         diff = prd_inv - grd_inv
#         inv_loss_tens = torch.abs(diff + F.softplus(-2.*diff) - log(2.))
#         inv_loss = inv_loss_tens.mean()
# 
#         tot_loss = self.con_weight*con_loss + self.bin_weight*bin_loss + self.inv_weight*inv_loss
# 
#         return tot_loss, con_loss.item(), bin_loss.item(), inv_loss.item()
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


# class ForkLoss(nn.Module):
#     def __init__(self,
#                  # bins: torch.tensor,
#                  pos_weight: float,
#                  con_weight: float,
#                  bin_weight: float,
#                  inv_weight: float,
#                  n_dists: int = 10):
#         super().__init__()
#         # assert 0.0 <= dist_weight <= 1.0
#         assert (con_weight + bin_weight + inv_weight) == 1.
# 
#         self.pos_weight = pos_weight
#         self.con_weight = con_weight
#         self.bin_weight = bin_weight
#         self.inv_weight = inv_weight
#         self.n_dists = n_dists
# 
# 
#     def forward(self, prd, grd):
#         prd_con, prd_bin, prd_inv = prd
#         grd_con, grd_bin, grd_inv = grd
# 
#         # contact loss
#         prd_con[grd_con.isnan()] = 0
#         grd_con[grd_con.isnan()] = 0
#         con_loss_tens = F.binary_cross_entropy(prd_con, grd_con, reduction="none")
#         con_loss_tens[grd_con == 0] *= self.pos_weight
#         con_loss_tens[grd_con == 1] *= 1 - self.pos_weight
#         con_loss = con_loss_tens.mean()
# 
#         # bin loss
#         prd_bin[grd_bin.isnan()] = 0
#         grd_bin[grd_bin.isnan()] = 0
#         bin_loss_tens = F.cross_entropy(prd_bin, grd_bin.argmax(1), reduction="none")
#         bin_loss = bin_loss_tens.mean()
# 
#         # inv loss
#         prd_inv[grd_inv.isnan()] = 0
#         grd_inv[grd_inv.isnan()] = 0
#         diff = prd_inv - grd_inv
#         inv_loss_tens = torch.abs(diff + F.softplus(-2.*diff) - log(2.))
#         inv_loss = inv_loss_tens.mean()
# 
#         tot_loss = self.con_weight*con_loss + self.bin_weight*bin_loss + self.inv_weight*inv_loss
# 
#         return tot_loss, con_loss.item(), bin_loss.item(), inv_loss.item()



class ForkLoss(nn.Module):
    def __init__(self,
                 # bins: torch.tensor,
                 pos_weight: float,
                 con_weight: float,
                 bin_weight: float,
                 inv_weight: float,
                 dists = None,
                 use_logit = False):
        super().__init__()
        # assert 0.0 <= dist_weight <= 1.0
        assert (con_weight + bin_weight + inv_weight) == 1.

        self.pos_weight = pos_weight
        self.con_weight = con_weight
        self.bin_weight = bin_weight
        self.inv_weight = inv_weight
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

        for ((key, prd_val), (_, grd_val)) in zip(prd["dists"].items(), grd["dists"].items()):
            loss_dict[key] = {}
            # bin loss
            mask = ~grd_val["bin"].isnan()[:, 0, ...]
            loss_tens = F.cross_entropy(prd_val["bin"], grd_val["bin"].argmax(1), reduction="none")
            loss_dict[key]["bin"] = loss_tens[mask].mean()
            tot_loss += self.bin_weight * loss_dict[key]["bin"]
            # inv loss
            diff = prd_val["inv"] - grd_val["inv"]
            loss_tens = torch.abs(diff + F.softplus(-2.*diff) - log(2.))
            loss_dict[key]["inv"] = 0.
            # loss_dict[key]["inv"] = loss_tens[:, mask].nanmean()
            # tot_loss += self.inv_weight * loss_dict[key]["inv"]

        loss_dict["tot"] = tot_loss 

        return loss_dict

