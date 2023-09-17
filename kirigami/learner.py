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


class KirigamiModule(pl.LightningModule):

    """
    Inherits from `pytorch_lightning.LightningModule` to create wrapper class
    for Kirigami network. Serves as main API to Kirigami pipeline.

    Attributes
    ----------
    threshold : float
        Pass
    optim : str
        Name of `torch.optim.Optimizer` object to be instantiated for training
        by `configure_optimizers`.
    lr : float
        Learning rate for training.
    model : kirigami.layers.ResNet
        Residual neural network module.
    crit : nn.Module
        Criterion for computing binary cross-entropy loss.
    post_proc : kirigami.layers.Greedy
        Post-processing module for enforcing constraints on output.
    raw_val_metrics : dict
        Dictionary of classification metrics for non-processed predicted
        labels on validation set as computed at each threshold in grid search.
    proc_val_metrics : dict
        Dictionary of classification metrics for post-processed predicted
        labels on validation set as computed at each threshold in grid search.
    n_val : int
        Number of molecules in validation set.
    test_rows : list
        List containing dictionary of classification metrics for each molecule
        in test set.

    Methods
    -------
    configure_optimizers()
        Sets up optimizer for training.
    on_fit_start()
        Logs code for Weights and Biases at start of training.
    training_step(batch, batch_idx)
        Script for step of training epoch.
    on_validation_epoch_start()
        Reinitializes validation metric caches for grid search at start of
        validation epoch.
    validation_step(batch, batch_idx)
        Script for step of validation epoch.
    on_validation_epoch_end()
        Perform grid search to determine optimal threshold for classification
        metrics.
    on_test_epoch_start()
        Reinitialize table of classification metrics for test set.
    test_step(batch, batch_idx)
        Script for step of test epoch.
    on_test_epoch_end
        Computes mean classification metrics at end of test epoch.
    forward(feat, post_proc=True)
        Performs single prediction step for input primary structure.
    """

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
        # forward pass
        self.n_val += 1
        feat, grd = batch
        prd = self.model(feat)
        prd = self.post_proc(prd, feat)
        loss = self.crit(prd, grd)
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
        # prd[prd < self.threshold.item()] = 0
        return prd
