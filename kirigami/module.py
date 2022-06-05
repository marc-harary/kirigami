import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class KirigamiModule(pl.LightningModule):
    def __init__(self, net, crit):
        super().__init__()
        self.net = net
        self.crit = crit

        self.val_mcc = torchmetrics.MatthewsCorrCoef(num_classes=2)
        for i in [1, 2, 5, 10]:
            setattr(self, f"pcc{i}", torchmetrics.PearsonCorrCoef())
            setattr(self, f"mae{i}", torchmetrics.MeanAbsoluteError())
        

    def training_step(self, batch, batch_idx):
        feat, lab_grd = batch
        lab_prd = self.net(feat)
        loss_dict = self.crit(lab_prd, lab_grd)
        self.log("train_loss",
                 bin_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss_dict["tot"]


    def validation_step(self, batch, batch_idx):
        feat, lab_grd = batch
        lab_prd = self.net(feat)
        loss_dict = self.crit(lab_prd, lab_grd)
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
