from pathlib import Path
import os
from math import ceil, floor
import math
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop


class DataModule(pl.LightningDataModule):

    dist_types_all = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]

    def __init__(self,
                 train_path: Path,
                 val_path: Path,
                 test_path: Path = None,
                 predict_path: Path = None,
                 batch_size: int = 1,
                 bin_min: float = None,
                 bin_max: float = None,
                 bin_step: float = None,
                 densify = False,
                 dists = None,
                 feats = None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        if bin_min is not None:
            self.bin_min = bin_min
            self.bin_max = bin_max
            self.bin_step = bin_step
            idx_min = math.floor(bin_min / bin_step + .5)
            idx_max = math.floor(bin_max / bin_step + .5)
            self.n_bins = idx_max - idx_min + 1 
        self.feats = feats if feats is not None else []
        self.dists = dists or []
        self.batch_size = batch_size
        self.densify = densify

        
    def setup(self, stage: str):
        for subset in ["train", "val", "test", "predict"]:
            path = getattr(self, subset + "_path")
            if path is None:
                continue
            dataset = torch.load(path)
            dataset = self._filt_dset(dataset)
            if self.densify:
                self._densify(dataset)
            setattr(self, subset + "_dataset", dataset)


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


    def predict_dataloader(self):
            return DataLoader(self.predict_dataset,
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


    # def _one_hot_bin(self, ipt):
    #     n_bins = self.bins.numel()
    #     n_data = ipt.numel()
    #     # expand both tensors to shape (ipt.size(), bins.size())
    #     ipt_flat = ipt.flatten()
    #     ipt_exp = ipt_flat.expand(n_bins, -1)
    #     bins_exp = self.bins.expand(n_data, -1).T
    #     # find which bins ipt fits in
    #     bin_bools = (ipt_exp <= bins_exp).int()
    #     # get index of largest bin ipt fits in
    #     vals, idxs = torch.max(bin_bools, 0)
    #     # if max is 0, then val is greater than every bin
    #     idxs[vals == 0] = n_bins - 1
    #     # construct one-hot
    #     one_hot = torch.zeros(n_bins, n_data)#, device=ipt.device)
    #     one_hot[idxs, torch.arange(n_data)] = 1
    #     # reshape back into ipt's shape
    #     one_hot = one_hot.reshape(-1, n_bins, ipt.shape[-2], ipt.shape[-1])
    #     one_hot[..., ipt.isnan()] = torch.nan
    #     return one_hot


    def _one_hot_bin(self, ipt):
        ipt = ipt.clip(self.bin_min, self.bin_max)
        idx_min = math.floor(self.bin_min / self.bin_step + .5)
        idx_max = math.floor(self.bin_max / self.bin_step + .5)
        num_classes = idx_max - idx_min + 1
        idx = torch.floor(ipt / self.bin_step + .5).long() - idx_min
        idx[..., ipt.isnan()] = 0 # need to get rid of nans for one_hot
        opt = F.one_hot(idx, num_classes).float()
        opt = opt.transpose(-1, 0)
        opt[..., ipt.isnan()] = torch.nan
        return opt


    def _pad(self, tens, L, val):
        tens_L = tens.shape[-1]
        diff = (L - tens_L) / 2
        padding = 2 * [floor(diff), ceil(diff)]
        return F.pad(tens, padding, "constant", val)


    def _collate_fn(self, batch):
        b_size = len(batch)
        max_L = max([row["seq"].shape[-1] for row in batch])
        feat = torch.zeros(b_size, len(self.feats)+8, max_L, max_L)
        # feat = torch.zeros(b_size, 8, max_L, max_L)
        lab = {}
        lab["con"] = torch.full((b_size, 1, max_L, max_L), torch.nan)
        lab["dists"] = {}

        for i, row in enumerate(batch):
            L = row["seq"].shape[-1]
            diff = (max_L - L) / 2
            start, end = floor(diff), floor(diff) + L
            if row["seq"].is_sparse:
                feat[i, :8, start:end, start:end] = self._concat(row["seq"].to_dense())
                # for j, feat_name in enumerate(self.feats):
                #     feat[i, 8+j, start:end, start:end] = row[feat_name].to_dense()
                #     feat[i, 8+j, start:end, start:end] = feat[i, 8+j, start:end, start:end] + feat[i, 8+j, start:end, start:end].transpose(-1, -2)
                lab["con"][i, :, start:end, start:end] = row["dssr"].to_dense()
            else:
                feat[i, :8, start:end, start:end] = self._concat(row["seq"])
                for j, feat_name in enumerate(self.feats):
                    feat[i, 8+j, start:end, start:end] = row[feat_name]
                    feat[i, 8+j, start:end, start:end] = feat[i, 8+j, start:end, start:end] + feat[i, 8+j, start:end, start:end].transpose(-1, -2)
                lab["con"][i, :, start:end, start:end] = row["dssr"]
        diag = torch.arange(max_L)
        lab["con"][..., diag, diag] = torch.nan
        
        for dist in self.dists:
            lab["dists"][dist] = {}
            lab["dists"][dist]["raw"] = torch.full((b_size, 1, max_L, max_L), torch.nan)
            lab["dists"][dist]["bin"] = torch.full((b_size, self.n_bins, max_L, max_L), torch.nan)

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

        # params = RandomCrop.get_params(lab["con"], (32, 32))
        # feat = crop(feat, *params)
        # lab["con"] = crop(lab["con"], *params)
        # for dist in self.dists:
        #     lab["dists"][dist]["raw"] = crop(lab["dists"][dist]["raw"], *params)
        #     lab["dists"][dist]["bin"] = crop(lab["dists"][dist]["bin"], *params)
        
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

