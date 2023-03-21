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
from torchmetrics.functional import f1_score
from torchmetrics import (MatthewsCorrCoef, F1Score, PrecisionRecallCurve,
    Precision, Recall, PearsonCorrCoef, MeanAbsoluteError)
from torchmetrics.functional.classification import * 

from kirigami.resnet import ResNet, ResNetParallel
from kirigami.post import Greedy, Dynamic, Symmetrize, RemoveSharp, Blossom
from kirigami.utils import mat2db
from kirigami.loss import ForkLoss



class KirigamiModule(pl.LightningModule):
    
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

        self.dists = [] if not dists else dists
        self.cutoffs = torch.linspace(0, 1, n_cutoffs)
        self.transfer = transfer
        self.bin_step = torch.nn.Parameter(torch.tensor([bin_step]),
                                           requires_grad=False)
        self.bin_min = torch.nn.Parameter(torch.tensor([bin_min]),
                                          requires_grad=False)
        self.bin_max = torch.nn.Parameter(torch.tensor([bin_max]),
                                          requires_grad=False)
        self.threshold = torch.nn.Parameter(torch.tensor([0.]),
                                           requires_grad=False)
        self.lr = lr
        self.T_max = T_max
        self.optim = getattr(torch.optim, optim)
        self.momentum = momentum
        self.prefix = "transfer" if self.transfer else "pre"
        
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

        if self.transfer:
            self.model = ResNetParallel(ipt_channels=ipt_channels,
                                        n_blocks=n_blocks,
                                        n_channels=n_channels,
                                        kernel_sizes=kernel_sizes,
                                        dilations=dilations,
                                        activation=activation,
                                        dropout=dropout)
        else:
            self.model = ResNet(ipt_channels=ipt_channels,
                                n_blocks=n_blocks,
                                n_channels=n_channels,
                                kernel_sizes=kernel_sizes,
                                dilations=dilations,
                                activation=activation,
                                dropout=dropout)

        self.crit = ForkLoss(pos_weight=pos_weight, con_weight=con_weight)
            
        if post_proc == "greedy":
            self.post_proc = Greedy()
        elif post_proc == "blossom":
            self.post_proc = Blossom()
        elif post_proc == "dynamic":
            self.post_proc = Dynamic()
        else:
            raise ValueError("Invalid post-processing type.")

    
    def forward(self, ipt, post_proc=True):
        self.model.eval()
        self.post_proc.eval()
        opt = self.model(ipt)
        if post_proc:
            opt["con"] = self.post_proc(opt["con"], ipt)
        return opt


    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code(".")


    def on_validation_epoch_start(self):
        self.on_training_epoch_start()
        for metric_name in self.metrics.keys():
            for prd_name in ["proc", "raw"]:
                setattr(self, f"{prd_name}_{metric_name}", [])
            

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
        # return loss
        return loss_dict["tot"]


    def on_validation_epoch_end(self):
        self.on_training_epoch_end()
        metrics = {}
        for metric in self.metrics.keys():
            metrics[metric] = torch.tensor(getattr(self, f"proc_{metric}")).float().mean(0)
        idx = metrics["mcc"].argmax()
        self.threshold[0] = self.metrics_thres[idx]
        self.log(f"{self.prefix}/val/proc/mcc", metrics["mcc"][idx], prog_bar=True)
        self.log(f"{self.prefix}/val/proc/threshold", self.threshold)
        for key in ["f1", "prec", "rec"]:
            self.log(f"{self.prefix}/val/proc/{key}", metrics[key][idx])
        

    def validation_step(self, batch, batch_idx):
        feat, lab_grd = batch
        lab_prd = self.model(feat)

        prd_raw = self.post_proc(lab_prd["con"], feat, sym_only=True)
        prd_proc = self.post_proc(lab_prd["con"], feat, sym_only=False)
        
        # mask out nan's
        grd = lab_grd["con"]
        mask = ~grd.isnan()
        grd = grd[mask]
        prd_raw = prd_raw[mask]
        prd_proc = prd_proc[mask]

        loss_dict = self.crit(lab_prd, lab_grd)

        self.log(f"{self.prefix}/val/tot_loss", loss_dict["tot"])
        self.log(f"{self.prefix}/val/raw/loss", F.binary_cross_entropy(prd_raw, grd.float()))
        self.log(f"{self.prefix}/val/proc/loss", F.binary_cross_entropy(prd_proc, grd.float()))

        # log metrics for raw and post-processed
        for metric_name, metric in self.metrics.items():
            for prd_name, prd in zip(["proc", "raw"], [prd_proc, prd_raw]):
                cur_metrics = []
                for thres in self.metrics_thres:
                    cur_metrics.append(metric(prd, grd.int(), threshold=thres.item()))
                metric_list = getattr(self, f"{prd_name}_{metric_name}")
                metric_list.append(cur_metrics)
    
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

        self.logger.log_table(key=f"{self.prefix}/val/mcc_cutoff", data=mccs_all,
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

        self.log(f"{self.prefix}/test/tot_loss", loss_dict["tot"])
        self.log(f"{self.prefix}/test/con_loss", loss_dict["con"])
        self.log(f"{self.prefix}/test/mcc", self.test_mcc)

        lab_prd["con"][lab_prd["con"] < self.test_mcc.threshold] = 0
        dbn = mat2db(lab_prd["con"])
        self.test_rows.append((dbn, test_mcc.item(), test_f1.item()))


    def on_test_epoch_end(self):
        self.logger.log_table(key=f"{self.prefix}/test/scores", data=self.test_rows,
            columns=["dbn", "mcc", "f1"])
        mccs = torch.tensor([row[1] for row in self.test_rows])
        f1s = torch.tensor([row[2] for row in self.test_rows])
        self.log(f"{self.prefix}/test/mcc_mean", mccs.mean().item())
        self.log(f"{self.prefix}/test/mcc_median", mccs.median().item())
        self.log(f"{self.prefix}/test/f1_mean", f1s.mean().item())
        self.log(f"{self.prefix}/test/f1_median", f1s.median().item())


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        feat, lab_grd = batch
        return self.model(feat)


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

    
    def _dequantize(self, ipt):
        idxs = ipt.argmax(-3).float()
        opt = idxs * self.bin_step
        opt += self.bin_min
        return opt 

