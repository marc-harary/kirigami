import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from kirigami.metrics import GroundMCC


class KirigamiModule(pl.LightningModule):
    
    Ls = [1, 2, 5, 10]
    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]
    
    def __init__(self, net, crit):
        super().__init__()
        self.net = net
        self.crit = crit
        # self.val_metrics = {}
        self.mcc = GroundMCC()
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
        lab_prd = self.net(feat)
        loss_dict = self.crit(lab_prd, lab_grd)
        self.log("train_tot_loss",
                 loss_dict["tot"],
                 on_epoch=True,
                 logger=True)
        self.log("train_con_loss",
                 loss_dict["con"],
                 on_epoch=True,
                 logger=True)
        return loss_dict["tot"]


    def validation_step(self, batch, batch_idx):
        feat, lab_grd = batch
        lab_prd = self.net(feat)
        loss_dict = self.crit(lab_prd, lab_grd)
        self.mcc(lab_prd, lab_grd, feat)
        self.log("val_tot_loss", loss_dict["tot"], on_epoch=True, logger=True)
        self.log("val_con_loss", loss_dict["con"], on_epoch=True, logger=True)
        self.log("val_mcc", self.mcc, on_epoch=True, logger=True)
        self.logger.log_image(key=f"mol{batch_idx:02}", images=[lab_grd["con"].squeeze(), lab_prd["con"].squeeze()])
        return loss_dict["tot"]


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def _format_dist(self, prd, grd):
        grd_vec = torch.triu(grd, diagonal=5)
        prd_vec = torch.triu(prd, diagonal=5)   
        prd_vec = prd_vec[~grd_vec.isnan()]
        grd_vec = grd_vec[~grd_vec.isnan()]
        return prd_vec, grd_vec
