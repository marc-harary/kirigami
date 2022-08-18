import sys
import os

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.cli import LightningCLI

from argparse import ArgumentParser

import wandb

from kirigami.data import DataModule
from kirigami.qrna import QRNANet
from kirigami.spot import ResNet
from kirigami.fork import Fork
from kirigami.loss import ForkLoss
from kirigami.learner import KirigamiModule
from kirigami.loop import CutoffLoop


def densify(dset):
    for row in dset:
        for key, value in row.items():
            if key == "dist":
                for key, value in row["dists"].items():
                    if row["dists"][key].is_sparse:
                        row["dists"][key] = value.to_dense()
            else:
                if row[key].is_sparse:
                    row[key] = row[key].to_dense()


feat_choices = {"pf", "pfold", "petfold", "pfile", "rnafold", "prob_pair",
                "centroid", "subopt0", "subopt1", "subopt2", "subopt3"} 


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--norm", type=str, default="InstanceNorm2d", choices=["InstanceNorm2d", "BatchNorm2d"])
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--feats", type=str, choices=feat_choices, nargs="+")
    parser.add_argument("--acc-grad", type=int, default=1)
    args = parser.parse_args()

    args.feats = list(set(args.feats))

    if not args.sync:
        os.environ["WANDB_MODE"] = "offline"

    bins = torch.arange(2, 21, .5)
    learner = KirigamiModule(n_blocks=32, 
                             n_channels=32,
                             p=.2,
                             arch="QRNA",
                             norm=args.norm,
                             chunks=8,
                             post_proc="greedy",
                             bins=bins,
                             pos_weight=.5,
                             con_weight=1.0,
                             inv_weight=0.0,
                             bin_weight=0.,
                             feats=args.feats,
                             dists=None,
                             cutoffs=torch.linspace(0., 1., 1000))

    train_dataset = torch.load("data/spot/SPOT-TR0-all.pt")
    val_dataset = torch.load("data/spot/VL1-all.pt")
    test_dataset = torch.load("data/spot/TS1-all.pt")

    densify(train_dataset)
    densify(val_dataset)
    densify(test_dataset)

    # torch.backends.cudnn.benchmark = True

    data_mod = DataModule(train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          feats=args.feats,
                          dists=None,
                          batch_size=args.batch_size,
                          bins=bins)

    wandb_logger = WandbLogger(project="kirigami")
    wandb_logger.experiment.log_code(".")

    # print("starting pre-training...")
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor="val_mcc", mode="max")],
                         max_time="01:18:00:00",
                         max_epochs=170,
                         logger=wandb_logger,
                         auto_lr_find=True,
                         accelerator="auto",
                         devices=-1,
                         strategy="ddp",
                         # precision=16,  
                         # amp_backend="native",
                         check_val_every_n_epoch=1,
                         accumulate_grad_batches=args.acc_grad)
    trainer.fit(learner, datamodule=data_mod)

    train_dataset = torch.load("data/spot/TR1-all.pt")
    densify(train_dataset)
    data_mod = DataModule(train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          feats=args.feats,
                          batch_size=args.batch_size,
                          dists=None,
                          bins=bins)

    # print("starting post-training...")
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor="val_mcc", mode="max")],
                         max_time="00:12:00:00",
                         max_epochs=2000,
                         logger=wandb_logger,
                         auto_lr_find=True,
                         accelerator="auto",
                         devices=-1,
                         strategy="ddp",
                         check_val_every_n_epoch=1,
                         amp_backend="native",
                         accumulate_grad_batches=args.acc_grad)
    trainer.fit(learner, datamodule=data_mod)
    trainer.test(learner, ckpt_path="best", datamodule=data_mod)
    

if __name__ == "__main__":
    main()
