import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics
from torchmetrics.functional import matthews_corrcoef
from torchmetrics.functional import f1_score
from torchmetrics import MatthewsCorrCoef, F1Score

from tqdm import tqdm

from kirigami.spot import ResNetBlock
from kirigami.qrna import QRNABlock
from kirigami.fork import Fork, PostProcess
from kirigami.utils import mat2db


class KirigamiModule(pl.LightningModule):
    
    Ls = [1, 2, 5, 10]
    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]
    
    def __init__(self,
                 n_blocks: int,
                 n_channels: int,
                 p: float,
                 use_spot: bool,
                 crit: nn.Module,
                 bins: torch.Tensor,
                 cutoffs: torch.Tensor):
        super().__init__()
        self.crit = crit
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=2, threshold=.5)
        # self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=2, threshold=1e-4)
        self.val_f1 = F1Score(threshold=1e-4)
        self.test_mcc = MatthewsCorrCoef(num_classes=2)
        self.test_f1 = F1Score(threshold=1e-4)
        self.cutoffs = cutoffs
        self.n_blocks = n_blocks

        conv_init = torch.nn.Conv2d(in_channels=10,
        # conv_init = torch.nn.Conv2d(in_channels=10,
                                    out_channels=n_channels,
                                    kernel_size=3,
                                    padding=1)

        dilations = 2 * n_blocks * [1]
        blocks = []
        if use_spot:
            for i in range(n_blocks):
                block = ResNetBlock(p=p,
                                    dilations=dilations[2*i:2*(i+1)],
                                    kernel_sizes=(3,5),
                                    n_channels=n_channels)
                blocks.append(block)
        else:
            for i in range(n_blocks):
                block = QRNABlock(p=p,
                                  dilations=dilations[2*i:2*(i+1)],
                                  kernel_sizes=(3,5),
                                  n_channels=n_channels,
                                  resnet=True)
                blocks.append(block)
        fork = Fork(n_channels=n_channels, n_bins=len(bins), kernel_size=5)
        self.model = torch.nn.Sequential(*[conv_init, *blocks, fork])
        self.post_proc = PostProcess()

        self.test_rows = []
        self.test_vals = []
        
        # self.val_metrics = {}
        # self.val_metrics["con"] = torchmetrics.MatthewsCorrCoef(num_classes=2)
        # self.val_metrics["dists"] = {}
        # for dist_type in self.dist_types:
        #     self.val_metrics["dists"][dist_type] = {}
        #     self.val_metrics["dists"][dist_type]["bin"] = {}
        #     self.val_metrics["dists"][dist_type]["inv"] = {}
        #     for i in self.Ls:
        #         self.val_metrics["dists"][dist_type]["bin"][f"pcc{i}"] = torchmetrics.PearsonCorrCoef()
        #         self.val_metrics["dists"][dist_type]["bin"][f"mae{i}"] = torchmetrics.PearsonCorrCoef()
        #         self.val_metrics["dists"][dist_type]["inv"][f"pcc{i}"] = torchmetrics.MeanAbsoluteError()
        #         self.val_metrics["dists"][dist_type]["inv"][f"mae{i}"] = torchmetrics.MeanAbsoluteError()
        

    def training_step(self, batch, batch_idx):
        feat, lab_grd = batch

        lab_prd = self.model(feat)
        # lab_prd["con"] = self.post_proc(lab_prd["con"], feat)
        loss_dict = self.crit(lab_prd, lab_grd)

        self.log("train_tot_loss", loss_dict["tot"], on_epoch=True, logger=True, batch_size=True)
        self.log("train_con_loss", loss_dict["con"], on_epoch=True, logger=True, batch_size=True)
        for dist in self.dist_types:
            self.log(f"train_{dist}_loss", loss_dict[dist]["bin"], on_epoch=True, logger=True, batch_size=True)

        return loss_dict["tot"]


    def validation_step(self, batch, batch_idx):
        feat, lab_grd = batch

        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat, torch.sum(lab_grd["con"] > 0).item() / 2)
        # lab_prd["con"] = self.post_proc(lab_prd["con"], feat)

        loss_dict = self.crit(lab_prd, lab_grd)
        self.mcc(lab_prd["con"], lab_grd["con"].int())
        self.val_f1(lab_prd["con"].flatten(), lab_grd["con"].int().flatten())

        self.log("val_tot_loss", loss_dict["tot"], on_epoch=True, logger=True, batch_size=True)
        self.log("val_con_loss", loss_dict["con"], on_epoch=True, logger=True, batch_size=True)
        self.log("val_mcc", self.mcc, on_epoch=True, logger=True, batch_size=True)
        self.log("val_f1", self.val_f1, on_epoch=True, logger=True, batch_size=True)
        for dist in self.dist_types:
            self.log(f"val_{dist}_loss", loss_dict[dist]["bin"], on_epoch=True, logger=True, batch_size=True)
        # self.logger.log_image(key=f"mol{batch_idx:02}", images=[lab_grd["con"].squeeze(), lab_prd["con"].squeeze()])
        # self.logger.log_image(key=f"mol{batch_idx:02}", images=[lab_grd["con"].squeeze(), lab_prd["con"].squeeze()])

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
                mccs[i] = matthews_corrcoef(prd, grd, num_classes=2, threshold=cutoff)
            mean_mcc = mccs.mean().item()
            mccs_all.append((cutoff, mean_mcc))
            if mean_mcc > best_mcc:
                best_mcc = mean_mcc
                best_cutoff = cutoff

        self.logger.log_table(key="val_mcc_cutoff", data=mccs_all, columns=["cutoff", "mcc"])
        self.log("best_cutoff", best_cutoff)
        self.cutoff = self.test_mcc.threshold = self.test_f1.threshold = best_cutoff
        

    def test_step(self, batch, batch_idx):
        feat, lab_grd = batch

        lab_prd = self.model(feat)
        lab_prd["con"] = self.post_proc(lab_prd["con"], feat)

        loss_dict = self.crit(lab_prd, lab_grd)
        # test_mcc = self.test_mcc(lab_prd["con"].triu(), lab_grd["con"].int().triu())
        test_mcc = self.mcc(lab_prd["con"], lab_grd["con"].int())
        test_f1 = self.test_f1(lab_prd["con"].flatten(), lab_grd["con"].int().flatten())

        self.log("test_tot_loss", loss_dict["tot"], on_epoch=True, logger=True, batch_size=True)
        self.log("test_con_loss", loss_dict["con"], on_epoch=True, logger=True, batch_size=True)
        self.log("test_mcc", self.test_mcc, on_epoch=True, logger=True, batch_size=True)

        dbn = mat2db(lab_prd["con"])
        self.test_rows.append((dbn, test_mcc.item(), test_f1.item()))
        # self.test_vals.append((con_, lab_prd["con"] > self.cutoff, lab_grd["con"], feat))
        # self.log("test_mcc", mcc, on_epoch=True, logger=True, batch_size=True)


    def on_test_end(self):
        self.logger.log_table(key="test_scores", data=self.test_rows, columns=["dbn", "mcc", "f1"])
        # self.logger.experiment.log({"predction": self.test_vals})


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def _format_dist(self, prd, grd):
        grd_vec = torch.triu(grd, diagonal=5)
        prd_vec = torch.triu(prd, diagonal=5)   
        prd_vec = prd_vec[~grd_vec.isnan()]
        grd_vec = grd_vec[~grd_vec.isnan()]
        return prd_vec, grd_vec
