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
from torchmetrics.functional.classification import * 

from kirigami.resnet import ResNet, ResNetParallel
from kirigami.post import Greedy, Dynamic, Symmetrize, RemoveSharp, Blossom
from kirigami.utils import mat2db, get_con_metrics
from kirigami.loss import ForkLoss



class KirigamiModule(pl.LightningModule):

    metrics = dict(mcc=binary_matthews_corrcoef,
                   f1=binary_f1_score,
                   prec=binary_precision,
                   rec=binary_recall) 
    
    def __init__(self,
                 ipt_channels: int,
                 n_blocks: int,
                 n_channels: int,
                 kernel_sizes: Tuple[int, int], 
                 dilations: Tuple[int, int],
                 activation: str,
                 dropout: float,
                 pos_weight: float,
                 con_weight: float,
                 transfer: bool,
                 optim: str,
                 lr: float,
                 T_max: int = None,
                 momentum: float = 0.9,
                 bin_step: float = torch.nan,
                 bin_min: float = torch.nan,
                 bin_max: float = torch.nan,
                 dists: list = None,
                 n_val_thres: int = 100, 
                 n_cutoffs: int = 1000,
                 post_proc: str = "greedy"):

        super().__init__()
        # non-trainable hyperparameters
        self.bin_step  = torch.nn.Parameter(torch.tensor([bin_step]), requires_grad=False)
        self.bin_min   = torch.nn.Parameter(torch.tensor([bin_min]),  requires_grad=False)
        self.bin_max   = torch.nn.Parameter(torch.tensor([bin_max]),  requires_grad=False)
        self.threshold = torch.nn.Parameter(torch.tensor([0.]),       requires_grad=False) # dummy value
        # training parameters
        self.dists = [] if not dists else dists
        self.transfer = transfer
        self.lr = lr
        self.T_max = T_max
        self.optim = getattr(torch.optim, optim)
        self.momentum = momentum
        self.prefix = "transfer" if self.transfer else "pretrain"
        self.metrics_thres = torch.linspace(0, 1, n_val_thres)
        # build network backbone
        if self.transfer:
            self.model = ResNetParallel(ipt_channels=ipt_channels,
                                        n_blocks=n_blocks,
                                        n_channels=n_channels,
                                        kernel_sizes=kernel_sizes,
                                        dilations=dilations,
                                        activation=activation,
                                        dropout=dropout)
            for dist in self.dist_types:
                setattr(self, f"val_{dist}_pcc", PearsonCorrCoef())
                setattr(self, f"val_{dist}_mae", MeanAbsoluteError())
        else:
            self.model = ResNet(ipt_channels=ipt_channels,
                                n_blocks=n_blocks,
                                n_channels=n_channels,
                                kernel_sizes=kernel_sizes,
                                dilations=dilations,
                                activation=activation,
                                dropout=dropout)
        # initialize criterion
        self.crit = ForkLoss(pos_weight=pos_weight, con_weight=con_weight)
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
        if self.optim is torch.optim.SGD:
            optimizer = self.optim(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = self.optim(self.parameters(), lr=self.lr)
        if self.T_max is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
            return [optimzer], [scheduler]
        else:
            return optimizer


    def on_fit_start(self):
        # needed b/c LightningCLI doesn't automatically log code
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code(".")


    def training_step(self, batch, batch_idx):
        # forward pass and compute loss
        feat, lab_grd = batch
        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
        loss_dict = self.crit(lab_prd, lab_grd)
        # log training metrics 
        self.log(f"{self.prefix}/train/tot_loss", loss_dict["tot"])
        self.log(f"{self.prefix}/train/con_loss", loss_dict["con"])
        for dist in self.dists:
            self.log(f"{self.prefix}/train/{dist}_bin_loss", loss_dict["dists"][dist])
        return loss_dict["tot"]


    def on_validation_epoch_start(self):
        self.raw_validation_metrics = dict(mcc=[], f1=[], precision=[], recall=[])
        self.proc_validation_metrics = dict(mcc=[], f1=[], precision=[], recall=[])


    def validation_step(self, batch, batch_idx):
        # forward pass
        feat, lab_grd = batch
        lab_prd = self.model(feat)
        prd_raw = self.post_proc(lab_prd["con"], feat, sym_only=True)
        prd_proc = lab_prd["con"] = self.post_proc(lab_prd["con"], feat, sym_only=False)
        # compute and log loss
        loss_dict = self.crit(lab_prd, lab_grd)
        self.log(f"{self.prefix}/val/tot_loss", loss_dict["tot"])
        self.log(f"{self.prefix}/val/proc/loss", loss_dict["proc"])
        # log metrics for post-processed prediction
        for thres in self.metrics_thres:
            metrics_dict = get_con_metrics(prd_proc, lab_grd["con"], thres.item())
            for key, val in metrics_dict.items():
                self.proc_validation_metrics[key].append(val)
        # log metrics for raw prediction
        self.log(f"{self.prefix}/val/raw/loss", F.binary_cross_entropy(prd_raw, grd.float()))
        for thres in self.metrics_thres:
            metrics_dict = get_con_metrics(prd_raw, lab_grd["con"], thres.item())
            for key, val in metrics_dict.items():
                self.raw_validation_metrics[key].append(val)
        
        # log distance metrics
        for dist in self.dists:
            self.log(f"{self.prefix}/val/{dist}_bin_loss", loss_dict["dists"][dist])
            # convert predicted to real-valued distance
            prd_real = lab_prd["dists"][dist]
            prd_real = self._dequantize(prd_real).flatten()
            # preprocess real-valued ground truth
            grd_real = lab_grd["dists"][dist]["raw"]
            grd_real = grd_real.clip(self.bin_min, self.bin_max)
            grd_real = grd_real.flatten()
            # mask nan's
            idxs = ~grd_real.isnan()
            prd_real = prd_real[idxs]
            grd_real = grd_real[idxs]
            # log PCC
            pcc_obj = getattr(self, f"val_{dist}_pcc")(prd_real, grd_real)
            pcc_obj(prd_real, grd_real)
            self.log(f"{self.prefix}/val/{dist}_pcc", pcc_obj)
            # log MAE
            mae_obj = getattr(self, f"val_{dist}_mae")
            mae_obj(prd_real, grd_real)
            self.log(f"{self.prefix}/val/{dist}_mae", mae_obj)
        return loss_dict["tot"]


    def on_validation_epoch_end(self):
        # compute sample-wise mean of all metrics
        mean_metrics = {}
        for key, val in self.raw_validation_metrics.items():
            mean = torch.tensor(val).float().mean(0)
            metrics[key] = mean
        # update threshold via gridsearch for max MCC
        idx = mean_metrics["mcc"].argmax()
        self.threshold[0] = self.metrics_thres[idx]
        # log key metrics
        self.log(f"{self.prefix}/val/proc/threshold", self.threshold.item())
        self.log(f"{self.prefix}/val/proc/mcc", metrics["mcc"][idx], prog_bar=True)
        for key in ["f1", "precision", "recall"]:
            self.log(f"{self.prefix}/val/proc/{key}", metrics[key][idx])


    def on_test_epoch_start(self):
        # just need to initialize output table
        self.test_rows = []
        

    def test_step(self, batch, batch_idx):
        # forward pass
        feat, lab_grd = batch
        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
        # compute and log loss
        loss_dict = self.crit(lab_prd, lab_grd)
        self.log(f"{self.prefix}/test/tot_loss", loss_dict["tot"])
        self.log(f"{self.prefix}/test/con_loss", loss_dict["con"])
        # convert to dbn, compute metrics,  and write row in table
        metrics_dict = get_con_metrics(lab_prd["con"], lab_grd["con"], threshold.item())
        prd[prd < self.threshold.item()] = 0
        dbn = mat2db(prd)
        self.test_rows.append((dbn, *metrics_dict.values()))


    def on_test_epoch_end(self):
        # log full output table all at once
        self.logger.log_table(key=f"{self.prefix}/test/scores",
                              data=self.test_rows,
                              columns=["dbn", "mcc", "f1"])
        # compute and log aggregate statistics for ease of vieiwing
        mccs = torch.tensor([row[1] for row in self.test_rows])
        f1s = torch.tensor([row[2] for row in self.test_rows])
        self.log(f"{self.prefix}/test/mcc_mean", mccs.mean().item())
        self.log(f"{self.prefix}/test/mcc_median", mccs.median().item())
        self.log(f"{self.prefix}/test/f1_mean", f1s.mean().item())
        self.log(f"{self.prefix}/test/f1_median", f1s.median().item())


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


    def _dequantize(self, ipt):
        idxs = ipt.argmax(-3).float()
        opt = idxs * self.bin_step
        opt += self.bin_min
        return opt 

