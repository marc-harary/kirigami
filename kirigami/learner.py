from typing import *

import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint_sequential
from torch.optim.lr_scheduler import *
from pytorch_lightning.loggers import WandbLogger

import torchmetrics
from torchmetrics.functional import matthews_corrcoef
from torchmetrics.functional.classification import (binary_matthews_corrcoef,
    binary_f1_score, binary_precision, binary_recall)

from kirigami.layers import * #ResNet, ResNetParallel
from kirigami.utils import mat2db, get_con_metrics



class KirigamiModule(pl.LightningModule):

    grid = torch.linspace(0, 1, 100)
    
    def __init__(self,
                 n_blocks: int,
                 n_channels: int,
                 kernel_sizes: Tuple[int, int], 
                 dilations: Tuple[int, int],
                 activation: str,
                 dropout: float,
                 optim: str,
                 lr: float,
                 post_proc: str = "greedy"):

        super().__init__()
        # non-trainable hyperparameters
        self.threshold = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False) # dummy value
        # training parameters
        self.optim = getattr(torch.optim, optim)
        self.lr = lr
        # build network backbone
        self.model = ResNet(n_blocks=n_blocks,
                            n_channels=n_channels,
                            kernel_sizes=kernel_sizes,
                            dilations=dilations,
                            activation=activation,
                            dropout=dropout)
        # initialize criterion
        self.crit = nn.BCELoss()
        # initialize post-processing module
        if post_proc == "greedy":
            self.post_proc = Greedy()
        elif post_proc == "blossom":
            self.post_proc = Blossom()
        elif post_proc == "dynamic":
            self.post_proc = Dynamic()
        else:
            raise ValueError("Invalid post-processing type.")

        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)#r, momentum=self.momentum)
        return optimizer


    def on_fit_start(self):
        # needed b/c LightningCLI doesn't automatically log code
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code(".")


    def training_step(self, batch, batch_idx):
        feat, grd = batch
        prd = self.model(feat)
        prd = self.post_proc(prd, feat)
        loss = self.crit(prd, grd)
        self.log("train/loss", loss)
        return loss


    def on_validation_epoch_start(self):
        self.raw_val_metrics = dict(mcc=torch.zeros_like(self.grid),
                                    f1=torch.zeros_like(self.grid),
                                    precision=torch.zeros_like(self.grid),
                                    recall=torch.zeros_like(self.grid))
        self.proc_val_metrics = dict(mcc=torch.zeros_like(self.grid),
                                     f1=torch.zeros_like(self.grid),
                                     precision=torch.zeros_like(self.grid),
                                     recall=torch.zeros_like(self.grid))
        self.n_val = 0


    def validation_step(self, batch, batch_idx):
        # forward pass
        feat, grd = batch
        prd = self.model(feat)
        prd_raw = self.post_proc(prd, feat, sym_only=True)
        prd_proc = self.post_proc(prd, feat, sym_only=False)
        # compute and log loss
        loss = self.crit(prd_proc, grd)
        self.log("val/loss", loss)
        self.log("val/raw/loss", self.crit(prd_raw, grd.float()))
        # log metrics
        for i, thres in enumerate(self.grid):
            proc_metrics = get_con_metrics(prd_proc, grd, thres.item())
            for key, val in proc_metrics.items():
                self.proc_val_metrics[key][i] += val
            raw_metrics = get_con_metrics(prd_raw, grd, thres.item())
            for key, val in raw_metrics.items():
                self.raw_val_metrics[key][i] += val
        self.n_val += 1
        return loss


    def on_validation_epoch_end(self):
        # update threshold via gridsearch for max MCC
        idx = self.proc_val_metrics["mcc"].argmax()
        self.threshold[0] = self.grid[idx]
        # log key metrics
        self.log("val/proc/threshold", self.threshold.item())
        for key in ["mcc", "f1", "precision", "recall"]:
            self.log(f"val/proc/{key}", self.proc_val_metrics[key][idx] / self.n_val,
                     prog_bar=(key=="mcc"))


    def on_test_epoch_start(self):
        # just need to initialize output table
        self.test_rows = []
        

    def test_step(self, batch, batch_idx):
        # forward pass
        feat, grd = batch
        prd = self.model(feat)
        prd = self.post_proc(prd, feat)
        # compute and log loss
        loss = self.crit(prd, grd)
        self.log("test/loss", loss)
        # convert to dbn, compute metrics, and write row in table
        metrics_dict = get_con_metrics(prd, grd, self.threshold.item())
        prd[prd < self.threshold.item()] = 0
        dbn = mat2db(prd)
        self.test_rows.append((dbn, *metrics_dict.values()))


    def on_test_epoch_end(self):
        # log full output table all at once
        self.logger.log_table(key="test/scores",
                              data=self.test_rows,
                              columns=["dbn", "mcc", "f1", "precision", "recall"])
        # compute and log aggregate statistics for ease of viewing
        mccs = torch.tensor([row[1] for row in self.test_rows])
        f1s = torch.tensor([row[2] for row in self.test_rows])
        self.log("test/mcc_mean", mccs.mean().item())
        self.log("test/mcc_median", mccs.median().item())
        self.log("test/f1_mean", f1s.mean().item())
        self.log("test/f1_median", f1s.median().item())


    def forward(self, ipt, post_proc=True):
        self.model.eval()
        self.post_proc.eval()
        opt = self.model(ipt)
        if post_proc:
            opt["con"] = self.post_proc(opt["con"], ipt)
        return opt


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        feat, lab_grd = batch
        return self.model(feat)

