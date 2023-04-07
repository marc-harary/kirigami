from pathlib import Path
import os
from math import ceil, floor
import math
from torch.nn import functional as F
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import zipfile
# import wget
from kirigami.utils import embed_st



class DataModule(pl.LightningDataModule):
    
    BPRNA_URL = "https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip"
    DATA_NAME = Path("bpRNA_dataset")

    def __init__(self, data_dir = "./"):
        super().__init__()
        self.data_dir = Path(data_dir)

        # self.train_path = train_path
        # self.val_path = val_path
        # self.test_path = test_path
        # self.predict_path = predict_path
        # if bin_min is not None:
        #     self.bin_min = bin_min
        #     self.bin_max = bin_max
        #     self.bin_step = bin_step
        #     idx_min = math.floor(bin_min / bin_step + .5)
        #     idx_max = math.floor(bin_max / bin_step + .5)
        #     self.n_bins = idx_max - idx_min + 1 
        # self.feats = feats if feats is not None else []
        # self.dists = dists or []
        # self.batch_size = batch_size
        # self.densify = densify


    def prepare_data(self):
        return
        # r = requests.get(self.BPRNA_URL, allow_redirects=True)
        # r = wget.download(self.BPRNA_URL)
        # f = open(self.data_dir / self.DATA_NAME.with_suffix(".zip"), "wb")
        # f.write(r.content)
        # f.close()

        # with zipfile.ZipFile(self.data_dir / self.DATA_NAME.with_suffix(".zip")) as f:
        #     f.extractall(self.DATA_NAME / self.data_dir)

        train_dir = self.DATA_NAME / self.data_dir / "TR0"
        files = list(train_dir.iterdir())
        files.sort()
        train_list = [embed_st(file) for file in tqdm(files)]
        torch.save(train_list, self.data_dir / "TR0.pt")

        val_dir = self.DATA_NAME / self.data_dir / "VL0"
        files = list(val_dir.iterdir())
        files.sort()
        val_list = [embed_st(file) for file in tqdm(files)]
        torch.save(val_list, self.data_dir / "VL0.pt")

        test_dir = self.DATA_NAME / self.data_dir / "TS0"
        files = list(test_dir.iterdir())
        files.sort()
        test_list = [embed_st(file) for file in tqdm(files)]
        torch.save(test_list, self.data_dir / "TS0.pt")
         

    def setup(self, stage: str):
        self.train_dataset = torch.load("TR0.pt")
        self.val_dataset = torch.load("VL0.pt")
        self.test_dataset = torch.load("TS0.pt")


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self._collate_fn,
                          shuffle=True,
                          pin_memory=True,
                          batch_size=1)


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


    def _collate_fn(self, batch):
        batch, = batch
        fasta, con = batch
        # length = fasta.shape[-1]
        # fasta = torch.kron(fasta, fasta)
        # fasta = fasta.reshape(length, length, 16)
        # fasta = fasta.transpose(-1, 0)
        # fasta = fasta[None, ...]
        fasta = fasta[..., None]
        fasta = torch.cat(fasta.shape[-2] * [fasta], dim=-1)
        fasta_t = fasta.transpose(-1, -2)
        fasta = torch.cat([fasta, fasta_t], dim=-3)
        fasta = fasta[None, ...]
        con = con[None, None, ...]
        return fasta.float(), con.float()

