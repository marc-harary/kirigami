from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from kirigami.layers import ResNet, Greedy
from kirigami.constants import GRID
from kirigami.utils import get_con_metrics


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
        optim: str,
        lr: float,
    ):
        super().__init__()
        # non-trainable hyperparameters
        self.threshold = torch.nn.Parameter(
            torch.tensor([0.0]), requires_grad=False
        )  # dummy value
        # training parameters
        self.optim = getattr(torch.optim, optim)
        self.lr = lr
        # build network backbone
        self.model = ResNet(
            n_blocks=n_blocks,
            n_channels=n_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            activation=activation,
            dropout=dropout,
        )
        # initialize criterion
        self.crit = nn.BCELoss()
        # initialize post-processing module
        self.post_proc = Greedy()

        self.raw_val_metrics = None
        self.proc_val_metrics = None
        self.n_val = None
        self.test_rows = None

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optim(
            self.parameters(), lr=self.lr
        )  # r, momentum=self.momentum)
        return optimizer

    def on_fit_start(self):
        # needed b/c LightningCLI doesn't automatically log code
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code(".")

    def training_step(self, batch, batch_idx):
        feat, grd = batch
        prd = self.model(feat)
        prd = self.post_proc(prd, feat)
        loss = self.crit(prd, grd)
        self.log("train/loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.raw_val_metrics = dict(
            mcc=torch.zeros_like(GRID),
            f1=torch.zeros_like(GRID),
            precision=torch.zeros_like(GRID),
            recall=torch.zeros_like(GRID),
        )
        self.proc_val_metrics = dict(
            mcc=torch.zeros_like(GRID),
            f1=torch.zeros_like(GRID),
            precision=torch.zeros_like(GRID),
            recall=torch.zeros_like(GRID),
        )
        self.n_val = 0

    def validation_step(self, batch, batch_idx):
        # forward pass
        feat, grd = batch
        prd = self.model(feat)
        prd_raw = self.post_proc(prd, feat, sym_only=True)
        prd_proc = self.post_proc(prd, feat, sym_only=False)
        # compute and log loss
        loss = self.crit(prd_proc, grd)
        self.log("val/loss", loss)
        self.log("val/raw/loss", self.crit(prd_raw, grd.float()))
        # log metrics
        for i, thres in enumerate(self.grid):
            proc_metrics = get_con_metrics(prd_proc, grd, thres.item())
            for key, val in proc_metrics.items():
                self.proc_val_metrics[key][i] += val
            raw_metrics = get_con_metrics(prd_raw, grd, thres.item())
            for key, val in raw_metrics.items():
                self.raw_val_metrics[key][i] += val
        self.n_val += 1
        return loss

    def on_validation_epoch_end(self):
        # update threshold via gridsearch for max MCC
        idx = self.proc_val_metrics["mcc"].argmax()
        self.threshold[0] = self.grid[idx]
        # log key metrics
        self.log("val/proc/threshold", self.threshold.item())
        for key in ["mcc", "f1", "precision", "recall"]:
            self.log(
                f"val/proc/{key}",
                self.proc_val_metrics[key][idx] / self.n_val,
                prog_bar=(key == "mcc"),
            )

    def on_test_epoch_start(self):
        # just need to initialize output table
        self.test_rows = []

    def test_step(self, batch, batch_idx):
        # forward pass
        feat, grd = batch
        prd = self.model(feat)
        prd = self.post_proc(prd, feat)
        loss = self.crit(prd, grd)
        self.log("test/loss", loss)
        metrics_dict = get_con_metrics(prd, grd, self.threshold.item())
        self.test_rows.append(metrics_dict)

    def on_test_epoch_end(self):
        # log full output table all at once
        if isinstance(self.logger, WandbLogger):
            self.logger.log_table(
                key="test/scores",
                data=self.test_rows,
                columns=["mcc", "f1", "precision", "recall"],
            )
        # compute and log aggregate statistics for ease of viewing
        mccs = torch.tensor([row["mcc"] for row in self.test_rows])
        f1s = torch.tensor([row["f1"] for row in self.test_rows])
        mccs[mccs.isnan()] = 0
        f1s[f1s.isnan()] = 0
        self.log("test/mcc_mean", mccs.mean().item())
        self.log("test/mcc_median", mccs.median().item())
        self.log("test/f1_mean", f1s.mean().item())
        self.log("test/f1_median", f1s.median().item())

    def forward(self, feat, post_proc=True):
        prd = self.model(feat)
        if post_proc:
            prd = self.post_proc(prd, feat)
        prd[prd < self.threshold.item()] = 0
        return prd
