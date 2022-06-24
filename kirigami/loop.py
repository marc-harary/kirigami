import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.loops import EvaluationLoop
from torchmetrics.functional import matthews_corrcoef as get_mcc
from torchmetrics import MatthewsCorrCoef


class CutoffLoop(EvaluationLoop):
    def __init__(self, cutoffs):
        super().__init__()
        self.cutoffs = cutoffs
        self.best_cutoff = 0.
        self.best_mcc = -torch.inf

    def on_run_start(self):
        net = self.trainer.lightning_module.net
        # check if network is on cuda
        device = next(net.parameters()).device
        net.eval()
        val_dataloader = self.trainer.datamodule.val_dataloader()
        mccs_all = []
        for cutoff in self.cutoffs: 
            mccs = [] 
            for feat, lab_grd in tqdm(val_dataloader):
                feat = feat.to(device)
                lab_prd = net(feat)
                con_ = lab_grd["con"].int()
                con_[lab_grd["con"].isnan()] = 0.
                con_ = con_.to(device)
                mcc = get_mcc(lab_prd["con"],
                              con_,
                              num_classes=2,
                              threshold=cutoff)
                mccs.append(mcc)
            mean_mcc = sum(mccs) / len(mccs)
            mccs_all.append((cutoff, mean_mcc))
            if mean_mcc > self.best_mcc:
                self.best_mcc = mean_mcc
                self.best_cutoff = cutoff
        mod.log_table(key="val_mcc_cutoff", data=mccs_all, columns=["cutoff", "mcc"])
        mod.cutoff = self.best_cutoff
        mod.test_mcc = MatthewsCorrCoef(num_classes=2, threshold=self.best_cutoff)
