import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint_sequential

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
from torchmetrics.functional.classification import 

from tqdm import tqdm

from kirigami.spot import ResNetBlock
from kirigami.qrna import QRNABlock
from kirigami.fork import Fork, ForkHead
from kirigami.post import Greedy, Dynamic, Symmetrize, RemoveSharp, Blossom
from kirigami.utils import mat2db
from kirigami.transformer import AttentionBlock
from kirigami.loss import ForkLoss



class KirigamiModule(pl.LightningModule):
    
    Ls = [1, 2, 5, 10]
    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4",
                  "O3O3", "NN"]
    
    def __init__(self,
                 n_blocks: int,
                 n_channels: int,
                 p: float,
                 feats: list, 
                 dists: list,
                 bins: torch.Tensor,
                 n_cutoffs: int,
                 pos_weight: float,
                 con_weight: float,
                 inv_weight: float,
                 bin_weight: float,
                 transfer: bool,
                 n_val_thres: int = 10, 
                 lr: float = 1e-5,
                 arch: str = "QRNA",
                 norm: str = "InstanceNorm2d",
                 chunks: int = 4,
                 post_proc: str = "greedy"):

        super().__init__()

        self.n_blocks = n_blocks
        self.chunks = chunks
        self.dists = [] if not dists else dists
        self.cutoffs = torch.linspace(0, 1, n_cutoffs)
        self.lr = lr
        self.transfer = transfer
        self.save_hyperparameters()

        self.metrics_thres = torch.linspace(0, 1, n_thres)
        self.metrics = dict(mcc=binary_matthews_corrcoef,
                            f1=binary_f1_score,
                            prec=binary_precision,
                            rec=binary_recall) 
        self.test_mcc = MatthewsCorrCoef(num_classes=2, threshold=0.5, task="binary"))
        self.test_f1 = F1Score(threshold=0.5, task="binary"))
        for dist in self.dist_types:
            setattr(self, f"val_{dist}_pcc", PearsonCorrCoef())
            setattr(self, f"val_{dist}_mae", MeanAbsoluteError())

        self.test_rows = []
        self.test_vals = []

        self.crit = ForkLoss(pos_weight=pos_weight,
                             con_weight=con_weight,
                             inv_weight=inv_weight,
                             bin_weight=bin_weight,
                             dists=dists,
                             use_logit=False)

        dilations = 2 * n_blocks * [1]
        conv_init = torch.nn.Conv2d(in_channels=len(feats) + 8,
                                    out_channels=n_channels,
                                    kernel_size=3,
                                    padding=1)
        blocks = []
        if arch == "SPOT":
            for i in range(n_blocks):
                block = ResNetBlock(p=p,
                                    dilations=dilations[2*i:2*(i+1)],
                                    kernel_sizes=(3,5),
                                    n_channels=n_channels)
                blocks.append(block)
                # blocks.append(Symmetrize())
                # blocks.append(RemoveSharp())
        elif arch == "QRNA":
            for i in range(n_blocks):
                block = QRNABlock(p=p,
                                  dilations=dilations[2*i:2*(i+1)],
                                  kernel_sizes=(3,5),
                                  n_channels=n_channels,
                                  norm=norm,
                                  resnet=True)
                blocks.append(block)
                # blocks.append(Symmetrize())
                # blocks.append(RemoveSharp())
        elif arch == "transformer":
            for i in range(n_blocks):
                block = AttentionBlock(dropout=p,
                                       embed_dim=n_channels,
                                       hidden_dim=n_channels,
                                       num_heads=8)
                blocks.append(block)
        fork = Fork(n_channels=n_channels, n_bins=len(bins), kernel_size=5, dists=dists, add_sigmoid=True)
        self.model = torch.nn.Sequential(*[conv_init, *blocks, fork])

        if post_proc == "greedy":
            self.post_proc = Greedy()
        else:
            self.post_proc = Dynamic()
        # self.post_proc = Blossom()




    def on_training_epoch_start(self):
        if self.trainer.precision == 32:
            return
        self.model[-1].add_sigmoid = False
        self.crit.use_logit = True


    def on_validation_epoch_start(self):
        self.on_training_epoch_start()
        for prd in ["proc", "raw"]:
            for metric in ["mcc", "f1", "prec", "rec"]:
                setattr(self, prd + "_" + metric, [])
            

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
            self.log(f"transfer/train/{dist}_bin_loss", loss_dict[dist]["bin"],
                on_epoch=True, logger=True, batch_size=True)
            self.log(f"transfer/train/{dist}_inv_loss", loss_dict[dist]["inv"],
                on_epoch=True, logger=True, batch_size=True)

        return loss_dict["tot"]


    def on_training_epoch_end(self):
        if self.trainer.precision == 32:
            return
        self.model[-1].add_sigmoid = True
        self.crit.use_logit = False


    def on_validation_epoch_end(self):
        self.on_training_epoch_end()
        mccs = torch.tensor(self.proc_mccs_)
        f1s = torch.tensor(self.proc_f1s_)
        recs = torch.tensor(self.proc_recs_)
        precs = torch.tensor(self.proc_precs_)
        idx = mccs.mean(0).argmax()
        thres = self.metrics_thres[idx]
        mcc = mccs.mean(0).max()
        prefix = "transfer" if self.transfer else "pre"
        self.log(f"{prefix}/val/proc/mcc", mcc, on_epoch=True, logger=True,
            prog_bar=True)
        self.log(f"{prefix}/val/proc/thres", thres, on_epoch=True,
            logger=True)
        self.log(f"{prefix}/val/proc/rec", recs.mean(0)[idx], on_epoch=True,
            logger=True)
        self.log(f"{prefix}/val/proc/prec", precs.mean(0)[idx], on_epoch=True,
            logger=True)
        

    def validation_step(self, batch, batch_idx):
        feat, lab_grd = batch
        lab_prd = self.model(feat)

        raw_con = lab_prd["con"]
        mask = ~lab_grd["con"].isnan()
        prd_raw = raw_con[mask]
        grd = lab_grd["con"].int()[mask]

        lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
        loss_dict = self.crit(lab_prd, lab_grd)
        prd_proc = lab_prd["con"][mask]

        prefix = "transfer" if self.transfer else "pre"
        self.log(f"{prefix}/val/tot_loss", loss_dict["tot"], logger=True)
        self.log(f"{prefix}/val/con_loss", loss_dict["con"], logger=True)

        for thres in self.metrics_thres:
            cur_metrics = []
            for metric in [b_mcc, b_f1, b_rec, b_prec]:
                for prd in [prd_proc, prd_grd]:
                    cur_metrics.append(metric(prd, grd, threshold=thres))
            self.metrics.append(cur_metrics)

        self.log(f"{prefix}/val/tot_loss", loss_dict["tot"], logger=True, batch_size=True)
        thres_str = int(thres * 100)
    
        for dist in self.dists:
            prd_real = lab_prd["dists"][dist]["bin"]
            prd_real = prd_real * self._get_mids().to(prd_real.device)
            prd_real = prd_real.sum(1).flatten()

            grd_real = lab_grd["dists"][dist]["raw"]
            grd_real = grd_real.clip(0, self.bins.max())
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
            self.log(f"transfer/val/{dist}_bin_loss", loss_dict[dist]["bin"],
                on_epoch=True, logger=True, batch_size=True)
            self.log(f"transfer/val/{dist}_inv_loss", loss_dict[dist]["inv"],
                on_epoch=True, logger=True, batch_size=True)
            
        return loss_dict["tot"]


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        feat, lab_grd = batch
        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat,
            torch.sum(lab_grd["con"] > 0).item() / 2)
        return lab_prd, lab_grd


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
                mccs[i] = matthews_corrcoef(prd, grd, num_classes=2,
                                            threshold=cutoff)
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
        test_mcc = self.test_mcc(lab_prd["con"], lab_grd["con"].int())
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


    def on_test_end(self):
        prefix = "transfer" if self.transfer else "pre"
        self.logger.log_table(key=f"{prefix}/test/scores", data=self.test_rows,
            columns=["dbn", "mcc", "f1"])


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def _format_dist(self, prd, grd):
        grd_vec = torch.triu(grd, diagonal=5)
        prd_vec = torch.triu(prd, diagonal=5)   
        prd_vec = prd_vec[~grd_vec.isnan()]
        grd_vec = grd_vec[~grd_vec.isnan()]
        return prd_vec, grd_vec


    def _add_dists(self, dists, opt_types = "both", kernel_size = 5):
        self.dists = dists
        self.model[-1].dists = dists
        for dist in dists:
            head = ForkHead(self.hparams["n_channels"],
                            len(self.hparams["bins"]),
                            kernel_size, opt_types)
            setattr(self.model[-1], dist, head)

    
    def _get_mids(self):
        mids_ = []
        mids_.append(.5 * self.bins[0])
        for i, val in enumerate(self.bins[:-1]):
            mids_.append(.5 * (val + self.bins[i+1]))
        mids = torch.tensor(mids_).reshape(1, -1, 1, 1)
        return mids

