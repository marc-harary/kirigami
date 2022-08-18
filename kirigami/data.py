import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.nn import functional as F
from math import ceil, floor


class DataModule(pl.LightningDataModule):

    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 bins: torch.Tensor,
                 batch_size: int = 1,
                 test_dataset = None,
                 dists = None,
                 feats = None,
                 inv_eps: float = 1e-8):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.bins = bins
        self.inv_eps = inv_eps
        self.feats = feats if feats is not None else []
        self.dists = dists

        # vec_lengths = np.array([int(row["seq"].shape[1]) for row in train_dataset])
        # lengths = set(vec_lengths)
        # self.train_idxs = []
        # for length in lengths:
        #     idxs = np.argwhere(vec_lengths == length)
        #     self.train_idxs.append(idxs.flatten().tolist())

        # self.val_idxs = list(range(len(val_dataset))) if val_dataset else None
        # self.test_idxs = list(range(len(test_dataset))) if test_dataset else None
        
        self.batch_size = batch_size


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self._collate_fn,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True,
                          batch_size=self.batch_size)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=self._collate_fn,
                          shuffle=False,
                          batch_size=1)


    def test_dataloader(self):
            return DataLoader(self.test_dataset,
                              collate_fn=self._collate_fn,
                              shuffle=False,
                              batch_size=1)


    def _concat(self, fasta):
        out = fasta.unsqueeze(-1)
        out = torch.cat(out.shape[-2] * [out], dim=-1)
        out_t = out.transpose(-1, -2)
        out = torch.cat([out, out_t], dim=-3)
        out = out.unsqueeze(0)
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
        one_hot = one_hot.reshape(ipt.shape[0], n_bins, ipt.shape[-2], ipt.shape[-1])
        return one_hot


    def _pad(self, tens, L, val):
        tens_L = tens.shape[-1]
        diff = (L - tens_L) / 2
        padding = 2 * [floor(diff), ceil(diff)]
        return F.pad(tens, padding, "constant", val)


    def _collate_row(self, row, pad_L):
        seq = self._concat(row["seq"])
        if seq.is_sparse:
            seq = seq.to_dense()
        feat_list = [seq]
        L = seq.shape[-1]
        for feat_name in self.feats:
            feat = row[feat_name].float()
            try:
                feat = feat.reshape_as(1, 1, L, L)
            except: # error handling accounts for what's probably a PyTorch bug
                # feat = torch.zeros(1, 1, L, L).to_sparse()
                feat = torch.zeros(1, 1, L, L)
            if feat.is_sparse:
                feat = feat.to_dense()
            feat_list.append(feat)
        feat = torch.cat(feat_list, 1).float()#.to_dense()
        feat = self._pad(feat, pad_L, 0.)

        pad = lambda tens: self._pad(tens, pad_L, torch.nan)

        lab = {}
        if row["dssr"].is_sparse:
            row["dssr"] = row["dssr"].to_dense()
        lab["con"] = row["dssr"].float().unsqueeze(0)
        lab["con"] = self._pad(lab["con"], pad_L, torch.nan)
        lab["dists"] = {}
        for dist_type, dist_ in zip(self.dist_types, row["dists"]):
            lab["dists"][dist_type] = {}
            if dist_.is_sparse:
                dist_ = dist_.to_dense()
            dist = dist_.unsqueeze(0).unsqueeze(0)
            dist[dist < 0] = torch.nan
            lab["dists"][dist_type]["raw"] = pad(dist)
            lab["dists"][dist_type]["inv"] = 1 / (dist + self.inv_eps)
            lab["dists"][dist_type]["inv"][dist <= 0] = torch.nan
            lab["dists"][dist_type]["inv"] = pad(lab["dists"][dist_type]["inv"])
            lab["dists"][dist_type]["bin"] = self._one_hot_bin(dist)
            # lab["dists"][dist_type]["bin"] = pad(self._one_hot_bin(dist))
            lab["dists"][dist_type]["bin"][:, :, dist_ <= 0] = torch.nan
            lab["dists"][dist_type]["bin"] = pad(lab["dists"][dist_type]["bin"])

        return feat, lab


    def _collate_fn(self, batch):
        b_size = len(batch)
        max_L = max([row["seq"].shape[-1] for row in batch])
        feat = torch.zeros(b_size, len(self.feats)+8, max_L, max_L)
        lab = {}
        lab["con"] = torch.full((b_size, 1, max_L, max_L), torch.nan)
        lab["dists"] = {}

        for i, row in enumerate(batch):
            L = row["seq"].shape[-1]
            diff = (max_L - L) / 2
            start, end = floor(diff), floor(diff) + L
            if row["seq"].is_sparse:
                feat[i, :8, start:end, start:end] = self._concat(row["seq"]).to_dense()
                for j, feat_name in enumerate(self.feats):
                    feat[i, 8+j, start:end, start:end] = row[feat_name].to_dense()
                lab["con"][i, :, start:end, start:end] = row["dssr"].to_dense()
                # lab["con"][start:end, start:end] = row["dssr"].to_dense()
            else:
                feat[i, :8, start:end, start:end] = self._concat(row["seq"])
                for j, feat_name in enumerate(self.feats):
                    feat[i, 8+j, start:end, start:end] = row[feat_name]
                lab["con"][i, :, start:end, start:end] = row["dssr"]
        # diag = torch.arange(max_L)
        # lab["con"][..., diag, diag] = torch.nan

        if self.dists is not None:
            for dist_type in self.dist_types:
                lab["dists"][dist_type] = {}
                lab["dists"][dist_type]["raw"] = torch.full((b_size, 1, max_L, max_L), torch.nan)
                lab["dists"][dist_type]["inv"] = torch.full((b_size, 1, max_L, max_L), torch.nan)
                lab["dists"][dist_type]["bin"] = torch.full((b_size, self.bins.numel(), max_L, max_L), torch.nan)

            for i, row in enumerate(batch):
                L = row["seq"].shape[-1]
                diff = (max_L - L) / 2
                start, end = floor(diff), floor(diff) + L
                for j, dist_type in enumerate(self.dist_types):
                    dist = row["dists"][j, ...]
                    if dist.is_sparse:
                        dist = dist.to_dense()
                    dist[dist < 0] = torch.nan
                    lab["dists"][dist_type]["raw"][i, 0, start:end, start:end] = dist

            for dist_type in self.dist_types:
                dist = lab["dists"][dist_type]["raw"]
                lab["dists"][dist_type]["bin"] = self._one_hot_bin(dist)
                lab["dists"][dist_type]["inv"] = 1 / (dist + self.inv_eps)
        return feat, lab

