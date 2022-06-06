import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl



class DataModule(pl.LightningDataModule):

    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 bins: torch.Tensor,
                 inv_eps: float = 1e-8):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.bins = bins
        self.inv_eps = inv_eps


    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self._collate_fn, shuffle=True, batch_size=1)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self._collate_fn, shuffle=False, batch_size=1)


    def _concat(self, fasta):
        out = fasta.unsqueeze(-1)
        out = torch.cat(out.shape[-2] * [out], dim=-1)
        out_t = out.transpose(-1, -2)
        out = torch.cat([out, out_t], dim=-3)
        return out


    def _one_hot_bin(self, ipt):
        n_bins = self.bins.numel()
        n_data = ipt.numel()
        # expand both tensors to shape (ipt.size(), bins.size())
        ipt_flat = ipt.flatten()
        ipt_exp = ipt_flat.expand(n_bins, -1)
        bins_exp = self.bins.expand(n_data, -1).T
        # find which bins ipt fits in
        bin_bools = (ipt_exp <= bins_exp).int()
        # get index of largest bin ipt fits in
        vals, idxs = torch.max(bin_bools, 0)
        # if max is 0, then val is greater than every bin
        idxs[vals == 0] = n_bins - 1
        # construct one-hot
        one_hot = torch.zeros(n_bins, n_data)#, device=ipt.device)
        one_hot[idxs, torch.arange(n_data)] = 1
        # reshape back into ipt's shape
        one_hot = one_hot.reshape(1, n_bins, ipt.shape[-1], ipt.shape[-1])
        return one_hot


    def _collate_fn(self, batch):
        seq_, pet_, dssr_, dists_ = batch[0] 
        # create feature tensor
        seq = self._concat(seq_.to_dense()).unsqueeze(0)
        try:
            pet = pet_.to_dense().unsqueeze(0).unsqueeze(0)
        except: # error handling accounts for what's probably a PyTorch bug
            pet = torch.zeros(1, 1, seq.shape[-1], seq.shape[-1])
        feat = torch.cat((seq, pet), 1).float()
        # create label dictionary
        lab = {}
        lab["con"] = dssr_.to_dense().float()
        # zero out diagonal
        diag = torch.arange(seq.shape[-1])
        lab["con"][..., diag, diag] = torch.nan
        lab["dists"] = {}
        for dist_type, dist_ in zip(self.dist_types, dists_):
            lab["dists"][dist_type] = {}
            dist = dist_.unsqueeze(0).unsqueeze(0)
            dist[dist < 0] = torch.nan
            lab["dists"][dist_type]["raw"] = dist
            lab["dists"][dist_type]["inv"] = 1 / (dist + self.inv_eps)
            lab["dists"][dist_type]["inv"][dist <= 0] = torch.nan
            lab["dists"][dist_type]["bin"] = self._one_hot_bin(dist)
            lab["dists"][dist_type]["bin"][:, :, dist_ <= 0] = torch.nan
        return feat, lab
