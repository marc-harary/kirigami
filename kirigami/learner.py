import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint_sequential
from torch.optim.lr_scheduler import *

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import wandb
import networkx as nx

import torchmetrics
from torchmetrics.functional import matthews_corrcoef
from torchmetrics.functional import f1_score
from torchmetrics import (MatthewsCorrCoef, F1Score, PrecisionRecallCurve,
    Precision, Recall, PearsonCorrCoef, MeanAbsoluteError)
from torchmetrics.functional.classification import * 

from tqdm import tqdm

from kirigami.resnet import ResNet, ResNetParallel, ConvNeXt # QRNABlock, ResNet
from kirigami.post import Greedy, Dynamic, Symmetrize, RemoveSharp, Blossom
from kirigami.utils import mat2db
from kirigami.loss import ForkLoss


METRICS = dict(f1=binary_f1_score,
               recall=binary_recall,
               precision=binary_precision,
               mcc=binary_matthews_corrcoef)


class KirigamiModule(pl.LightningModule):
    
    Ls = [1, 2, 5, 10]
    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4",
                  "O3O3", "NN"]
    
    def __init__(self,
                 model_kwargs: dict,
                 crit_kwargs: dict,
                 transfer: bool,
                 optim: str,
                 lr: float,
                 momentum: float = 0.9,
                 bin_step: float = None,
                 bin_min: float = None,
                 bin_max: float = None,
                 dists: list = None,
                 n_val_thres: int = 100, 
                 n_cutoffs: int = 1000,
                 post_proc: str = "greedy"):

        super().__init__()

        self.dists = [] if not dists else dists
        self.cutoffs = torch.linspace(0, 1, n_cutoffs)
        self.transfer = transfer
        self.bin_step = bin_step
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.lr = lr
        self.optim = getattr(torch.optim, optim)
        self.momentum = momentum

        self.save_hyperparameters()

        self.metrics_thres = torch.linspace(0, 1, n_val_thres)
        self.metrics = dict(mcc=binary_matthews_corrcoef,
                            f1=binary_f1_score,
                            prec=binary_precision,
                            rec=binary_recall) 
        self.test_mcc = MatthewsCorrCoef(num_classes=2, threshold=0.5, task="binary")
        self.test_f1 = F1Score(threshold=0.5, task="binary")
        if transfer:
            for dist in self.dist_types:
                setattr(self, f"val_{dist}_pcc", PearsonCorrCoef())
                setattr(self, f"val_{dist}_mae", MeanAbsoluteError())

        self.test_rows = []
        self.test_vals = []

        self.crit = ForkLoss(**crit_kwargs)

        if self.transfer:
            self.model = ResNetParallel(**model_kwargs)
        else:
            self.model = ResNet(**model_kwargs)
            
        if post_proc == "greedy":
            self.post_proc = Greedy()
        elif post_proc == "blossom":
            self.post_proc = Blossom()
        elif post_proc == "dynamic":
            self.post_proc = Dynamic()
        else:
            raise ValueError("Invalid post-processing type.")


    def on_training_epoch_start(self):
        if self.trainer.precision == 32:
            return
        self.model[-1].add_sigmoid = False
        self.crit.use_logit = True


    def on_validation_epoch_start(self):
        self.on_training_epoch_start()
        for metric_name in METRICS.keys():
            for prd_name in ["proc", "raw"]:
                setattr(self, f"{prd_name}_{metric_name}", [])
            

    def training_step(self, batch, batch_idx):
        feat, lab_grd = batch

        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
        loss_dict = self.crit(lab_prd, lab_grd)
    
        prefix = "transfer" if self.transfer else "pre"
        self.log(f"{prefix}/train/tot_loss", loss_dict["tot"], on_epoch=True,
            logger=True, batch_size=True)
        self.log(f"{prefix}/train/con_loss", loss_dict["con"], on_epoch=True,
            logger=True, batch_size=True)
        for dist in self.dists:
            self.log(f"transfer/train/{dist}_bin_loss", loss_dict["dists"][dist],
                on_epoch=True, logger=True, batch_size=True)

        return loss_dict["tot"]


    def on_training_epoch_end(self):
        if self.trainer.precision == 32:
            return
        self.model[-1].add_sigmoid = True
        self.crit.use_logit = False


    def on_validation_epoch_end(self):
        self.on_training_epoch_end()
        metrics = {}
        for metric in METRICS.keys():
            metrics[metric] = torch.tensor(getattr(self, f"proc_{metric}")).float().mean(0)
        idx = metrics["mcc"].argmax()
        thres = self.metrics_thres[idx]
        prefix = "transfer" if self.transfer else "pre"
        self.log(f"{prefix}/val/proc/mcc", metrics["mcc"][idx], on_epoch=True, logger=True,
            prog_bar=True)
        self.log(f"{prefix}/val/proc/threshold", thres, on_epoch=True,
            logger=True)
        for key in ["f1", "precision", "recall"]:
            self.log(f"{prefix}/val/proc/{key}", metrics[key][idx], on_epoch=True,
                logger=True)
        

    def validation_step(self, batch, batch_idx):
        feat, lab_grd = batch
        lab_prd = self.model(feat)

        prd_raw = self.post_proc(lab_prd["con"], feat, sym_only=True)
        prd_proc = self.post_proc(lab_prd["con"], feat, sym_only=False)

        grd = lab_grd["con"]
        mask = ~grd.isnan()
        grd = grd[mask]
        prd_raw = prd_raw[mask]
        prd_proc = prd_proc[mask]

        loss_dict = self.crit(lab_prd, lab_grd)

        prefix = "transfer" if self.transfer else "pre"
        self.log(f"{prefix}/val/tot_loss", loss_dict["tot"])
        self.log(f"{prefix}/val/raw/loss", F.binary_cross_entropy(prd_raw, grd.float()))
        self.log(f"{prefix}/val/proc/loss", F.binary_cross_entropy(prd_proc, grd.float()))

        for metric_name, metric in METRICS.items():
            for prd_name, prd in zip(["proc", "raw"], [prd_proc, prd_raw]):
                cur_metrics = []
                for thres in self.metrics_thres:
                    cur_metrics.append(metric(prd, grd.int(), threshold=thres.item()))
                metric_list = getattr(self, f"{prd_name}_{metric_name}")
                metric_list.append(cur_metrics)
    
        for dist in self.dists:
            prd_real = lab_prd["dists"][dist]
            prd_real = self._dequantize(prd_real).flatten()

            grd_real = lab_grd["dists"][dist]["raw"]
            grd_real = grd_real.clip(self.bin_min, self.bin_max)
            grd_real = grd_real.flatten()

            idxs = ~grd_real.isnan()
            prd_real = prd_real[idxs]
            grd_real = grd_real[idxs]

            pcc_obj = getattr(self, f"val_{dist}_pcc")
            pcc_obj(prd_real, grd_real)

            mae_obj = getattr(self, f"val_{dist}_mae")
            mae_obj(prd_real, grd_real)

            self.log(f"transfer/val/{dist}_pcc", pcc_obj, on_epoch=True,
                logger=True, batch_size=True)
            self.log(f"transfer/val/{dist}_mae", mae_obj, on_epoch=True,
                logger=True, batch_size=True)
            self.log(f"transfer/val/{dist}_bin_loss", loss_dict["dists"][dist],
                on_epoch=True, logger=True, batch_size=True)
            
        return loss_dict["tot"]


    def on_test_start(self):
        self.model.eval()
        self.post_proc.eval()
        # check if network is on cuda
        device = next(self.model.parameters()).device
        val_dataloader = self.trainer.datamodule.val_dataloader()

        # cache all predictions
        prd_grds = []
        for feat, lab_grd in tqdm(val_dataloader):
            feat = feat.to(device)
            lab_prd = self.model(feat)
            lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
            prd_con = lab_prd["con"]
            grd_con = lab_grd["con"]
            grd_con[grd_con.isnan()] = 0.
            grd_con = grd_con.int()
            grd_con = grd_con.to(device)
            prd_grds.append((prd_con, grd_con))

        # calculate MCC at successive cutoffs and store best
        mccs_all = []
        best_mcc = -torch.inf
        best_cutoff = -torch.inf
        for cutoff in tqdm(self.cutoffs): 
            mccs = torch.zeros(len(prd_grds), device=device)
            for i, (prd, grd) in enumerate(prd_grds):
                mccs[i] = binary_matthews_corrcoef(prd, grd, threshold=cutoff.item())
            mean_mcc = mccs.mean().item()
            mccs_all.append((cutoff, mean_mcc))
            if mean_mcc > best_mcc:
                best_mcc = mean_mcc
                best_cutoff = cutoff

        prefix = "transfer" if self.transfer else "pre"
        self.logger.log_table(key=f"{prefix}/val/mcc_cutoff", data=mccs_all,
            columns=["cutoff", "mcc"])
        self.log("best_cutoff", best_cutoff)
        self.cutoff = self.test_mcc.threshold = self.test_f1.threshold = best_cutoff
        

    def test_step(self, batch, batch_idx):
        feat, lab_grd = batch

        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
        loss_dict = self.crit(lab_prd, lab_grd)

        grd_con = lab_grd["con"]
        grd_con[grd_con.isnan()] = 0.
        test_mcc = self.test_mcc(lab_prd["con"], grd_con.int())
        test_f1 = self.test_f1(lab_prd["con"].flatten(), lab_grd["con"].int().flatten())

        prefix = "transfer" if self.transfer else "pre"
        self.log(f"{prefix}/test/tot_loss", loss_dict["tot"], on_epoch=True,
            logger=True, batch_size=True)
        self.log(f"{prefix}/test/con_loss", loss_dict["con"], on_epoch=True,
            logger=True, batch_size=True)
        self.log(f"{prefix}/test/mcc", self.test_mcc, on_epoch=True,
            logger=True, batch_size=True)

        lab_prd["con"][lab_prd["con"] < self.test_mcc.threshold] = 0
        dbn = mat2db(lab_prd["con"])
        self.test_rows.append((dbn, test_mcc.item(), test_f1.item()))


    def on_test_epoch_end(self):
        prefix = "transfer" if self.transfer else "pre"
        self.logger.log_table(key=f"{prefix}/test/scores", data=self.test_rows,
            columns=["dbn", "mcc", "f1"])
        mccs = torch.tensor([row[1] for row in self.test_rows])
        f1s = torch.tensor([row[2] for row in self.test_rows])
        self.log(f"{prefix}/test/mcc_mean", mccs.mean().item())
        self.log(f"{prefix}/test/mcc_median", mccs.median().item())
        self.log(f"{prefix}/test/f1_mean", f1s.mean().item())
        self.log(f"{prefix}/test/f1_median", f1s.median().item())


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        feat, lab_grd = batch
        return self.model(feat)


    def configure_optimizers(self):
        if self.optim is torch.optim.SGD:
            optimizer = self.optim(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            print("here")
            optimizer = self.optim(self.parameters(), lr=self.lr)
        return optimizer

    
    def _dequantize(self, ipt):
        idxs = ipt.argmax(-3).float()
        opt = idxs * self.bin_step
        opt += self.bin_min
        return opt 

