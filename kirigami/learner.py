from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from kirigami.layers import ResNet, Greedy
from kirigami.constants import GRID
from kirigami.utils import get_con_metrics
import kirigami
from kirigami.utils import _embed_fasta, mat2db, outer_concat


class KirigamiModule(pl.LightningModule):
    def __init__(
        self,
        n_blocks: int,
        n_channels: int,
        kernel_sizes: Tuple[int, int],
        dilations: Tuple[int, int],
        activation: str,
        dropout: float,
    ):
        super().__init__()
        # non-trainable hyperparameters
        self.threshold = torch.nn.Parameter(
            torch.tensor([0.0]), requires_grad=False
        )  # dummy value
        self.model = ResNet(
            n_blocks=n_blocks,
            n_channels=n_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            activation=activation,
            dropout=dropout,
        )
        self.post_proc = Greedy()
        self.save_hyperparameters()

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code(".")

    def training_step(self, batch, batch_idx):
        feat, grd = batch
        prd = self.model(feat)
        prd_proc = self.post_proc(prd, feat, sym_only=False)
        proc_metrics = get_con_metrics(prd_proc, grd)
        loss = F.binary_cross_entropy(prd, grd)
        self.log("train/loss", loss)
        self.log("train/pcc", proc_metrics["pcc"])
        return loss

    def validation_step(self, batch, batch_idx):
        feat, grd = batch
        prd = self.model(feat)
        prd_proc = self.post_proc(prd, feat, sym_only=False)
        proc_metrics = get_con_metrics(prd_proc, grd)
        loss = F.binary_cross_entropy(prd, grd)
        self.log("bpRNA_new/proc/loss", loss)
        self.log("bpRNA_new/proc/pcc", proc_metrics["pcc"])
        return loss

    def on_test_epoch_start(self):
        # just need to initialize output table
        self.test_rows = []
        self.mcc_grid = torch.zeros_like(GRID)
        self.n_val = 0

    def test_step(self, batch, batch_idx):
        self.n_val += 1
        feat, grd = batch
        prd = self.model(feat)
        prd = self.post_proc(prd, feat)
        loss = F.binary_cross_entropy(prd, grd)
        self.log("test/loss", loss)
        for i, val in enumerate(GRID):
            metrics_dict = get_con_metrics(prd, grd, val.item())
            self.mcc_grid[i] += metrics_dict["mcc"]  # / self.n_val

    def on_test_epoch_end(self):
        self.thres = GRID[self.mcc_grid.argmax()]
        self.log("test/thres", self.thres)
        self.log("test/mcc", self.mcc_grid.max() / self.n_val)

    def forward(self, feat, post_proc=True):
        prd = self.model(feat)
        if post_proc:
            prd = self.post_proc(prd, feat)
        prd[prd < self.threshold.item()] = 0
        return prd

    def __call__(self, seq):
        fasta = _embed_fasta(seq.upper())
        fasta = outer_concat(fasta)
        prd = self.forward(fasta)
        dbn = mat2db(prd)
        return dbn
