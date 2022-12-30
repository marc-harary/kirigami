from pathlib import Path
import os
from math import ceil, floor
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):

    dist_types_all = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]

    def __init__(self,
                 train_path: Path,
                 val_path: Path,
                 bins: torch.Tensor,
                 batch_size: int = 1,
                 test_path: Path = None,
                 densify = False,
                 dists = None,
                 feats = None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.bins = bins
        self.feats = feats if feats is not None else []
        self.dists = dists
        self.batch_size = batch_size
        self.densify = densify

    
    def setup(self, stage: str):
        train_dataset = torch.load(self.train_path)
        val_dataset = torch.load(self.val_path)
        test_dataset = torch.load(self.test_path)

        train_dataset = self._filt_dset(train_dataset)
        val_dataset = self._filt_dset(val_dataset)
        test_dataset = self._filt_dset(test_dataset)

        if self.densify:
            self._densify(train_dataset)
            self._densify(val_dataset)
            self._densify(test_dataset)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


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
        one_hot = one_hot.reshape(-1, n_bins, ipt.shape[-2], ipt.shape[-1])
        one_hot[..., ipt.isnan()] = torch.nan
        return one_hot


    def _pad(self, tens, L, val):
        tens_L = tens.shape[-1]
        diff = (L - tens_L) / 2
        padding = 2 * [floor(diff), ceil(diff)]
        return F.pad(tens, padding, "constant", val)


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
            else:
                feat[i, :8, start:end, start:end] = self._concat(row["seq"])
                for j, feat_name in enumerate(self.feats):
                    feat[i, 8+j, start:end, start:end] = row[feat_name]
                lab["con"][i, :, start:end, start:end] = row["dssr"]
        diag = torch.arange(max_L)
        lab["con"][..., diag, diag] = torch.nan

        for dist in self.dists:
            lab["dists"][dist] = {}
            lab["dists"][dist]["raw"] = torch.full((b_size, 1, max_L, max_L), torch.nan)
            lab["dists"][dist]["bin"] = torch.full((b_size, self.bins.numel(), max_L, max_L), torch.nan)

            for i, row in enumerate(batch):
                L = row["seq"].shape[-1]
                diff = (max_L - L) / 2
                start, end = floor(diff), floor(diff) + L
                dist_tensor = row["dists"][dist]
                if dist_tensor.is_sparse:
                    dist_tensor = dist_tensor.to_dense()
                dist_tensor[dist_tensor <= 0] = torch.nan
                lab["dists"][dist]["raw"][i, 0, start:end, start:end] = dist_tensor
                lab["dists"][dist]["bin"][i, :, start:end, start:end] = self._one_hot_bin(dist_tensor)

        return feat, lab

    
    def _densify(self, dset):
        for row in dset:
            for key, value in row.items():
                if key == "dist" or key == "dists":
                    for key, value in row["dists"].items():
                        if row["dists"][key].is_sparse:
                            row["dists"][key] = value.to_dense()
                else:
                    if row[key].is_sparse:
                        row[key] = row[key].to_dense()


    def _filt_dset(self, dset):# , feats, dists = None):
        out = []
        for row in dset:
            out_row = {}
            out_row["seq"] = row["seq"]
            out_row["dssr"] = row["dssr"]
            for feat in self.feats:
                out_row[feat] = row[feat]
            out_row["dists"] = {}
            for dist in self.dists:
                i = self.dist_types_all.index(dist)
                out_row["dists"][dist] = row["dists"][i]
            out.append(out_row)
        return out

