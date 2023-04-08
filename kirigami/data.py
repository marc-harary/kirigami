from pathlib import Path
from typing import *
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
from kirigami.utils import embed_st, read_fasta, embed_fasta


class DataModule(pl.LightningDataModule):
    BPRNA_URL = "https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip"
    DATA_NAME = Path("bpRNA_dataset")

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        predict_ipt_dir: Optional[Path] = None,
        predict_opt_dir: Optional[Path] = None,
    ):
        super().__init__()

        self.data_dir = data_dir or Path.cwd() / "data"
        self.predict_ipt_dir = predict_ipt_dir or self.data_dir / "predict_ipt"
        self.predict_opt_dir = predict_opt_dir or self.data_dir / "predict_opt"

        self.data_dir.exists() or self.data_dir.mkdir()
        self.predict_ipt_dir.exists() or self.predict_ipt_dir.mkdir()
        self.predict_opt_dir.exists() or self.predict_opt_dir.mkdir()

        self.train_path = self.data_dir / "TR0.pt"
        self.val_path = self.data_dir / "VL0.pt"
        self.test_path = self.data_dir / "TS0.pt"
        self.predict_path = self.data_dir / "predict.pt"

    def prepare_data(self):
        # r = requests.get(self.BPRNA_URL, allow_redirects=True)
        # r = wget.download(self.BPRNA_URL)
        # f = open(self.data_dir / self.DATA_NAME.with_suffix(".zip"), "wb")
        # f.write(r.content)
        # f.close()

        # with zipfile.ZipFile(self.data_dir / self.DATA_NAME.with_suffix(".zip")) as f:
        #     f.extractall(self.DATA_NAME / self.data_dir)

        if not self.train_path.exists():
            train_dir = self.DATA_NAME / self.data_dir / "TR0"
            files = list(train_dir.iterdir())
            files.sort()
            train_list = [embed_st(file) for file in tqdm(files)]
            torch.save(train_list, self.train_path)

        if not self.val_path.exists():
            val_dir = self.DATA_NAME / self.data_dir / "VL0"
            files = list(val_dir.iterdir())
            files.sort()
            val_list = [embed_st(file) for file in tqdm(files)]
            torch.save(val_list, self.val_path)

        if not self.test_path.exists():
            test_dir = self.DATA_NAME / self.data_dir / "TS0"
            files = list(test_dir.iterdir())
            files.sort()
            test_list = [embed_st(file) for file in tqdm(files)]
            torch.save(test_list, self.test_path)

        files = list(self.predict_ipt_dir.iterdir())
        files.sort()
        mols, fastas, feats = [], [], []
        for file in files:
            mol = file.stem
            fasta = read_fasta(file)
            feat = (embed_fasta(fasta),)
            fastas.append(fasta)
            feats.append(feat)
            mols.append(mol)
        torch.save((mols, fastas, feats), self.predict_path)

    def setup(self, stage: str):
        if stage == "train":
            self.train_dataset = torch.load(self.train_path)
        elif stage == "val":
            self.val_dataset = torch.load(self.val_path)
        elif stage == "test":
            self.test_dataset = torch.load(self.test_path)
        elif stage == "predict":
            self.predict_mols, self.predict_fastas, self.predict_dataset = torch.load(
                self.predict_path
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
            batch_size=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=self._collate_fn, shuffle=False, batch_size=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, collate_fn=self._collate_fn, shuffle=False, batch_size=1
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=self._collate_fn,
            shuffle=False,
            batch_size=1,
        )

    def _collate_fn(self, batch):
        (batch,) = batch
        fasta = batch[0]
        fasta = fasta[..., None]
        fasta = torch.cat(fasta.shape[-2] * [fasta], dim=-1)
        fasta_t = fasta.transpose(-1, -2)
        fasta = torch.cat([fasta, fasta_t], dim=-3)
        fasta = fasta[None, ...]
        fasta = fasta.float()
        if len(batch) == 1:
            return fasta
        else:
            con = batch[1]
            con = con[None, None, ...]
            con = con.float()
            return fasta, con
