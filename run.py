import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from kirigami.data import DataModule
from kirigami.qrna import QRNANet
from kirigami.spot import ResNet
from kirigami.fork import Fork
from kirigami.loss import ForkLoss
from kirigami.learner import KirigamiModule



def main():
    bins = torch.arange(2, 21, .5)
    main_net = ResNet(n_blocks=32, in_channels=9, n_channels=32)
    fork = Fork(n_channels=32, n_bins=len(bins), kernel_size=5)
    net = nn.Sequential(main_net, fork)

    crit = ForkLoss(pos_weight=.9, con_weight=1, inv_weight=0.0, bin_weight=0.0)

    train_dataset = torch.load("data/SPOT-TR1.pt")
    val_dataset = torch.load("data/SPOT-VL1.pt")
    data_mod = DataModule(train_dataset, val_dataset, bins)

    learner = KirigamiModule(net, crit)
    wandb_logger = WandbLogger(project="test-project")
    wandb_logger.experiment.log_code(".")
    trainer = pl.Trainer(max_epochs=1000, logger=wandb_logger, auto_lr_find=True, accelerator="auto", accumulate_grad_batches=32)
    trainer.fit(learner, data_mod)
    

if __name__ == "__main__":
    main()
