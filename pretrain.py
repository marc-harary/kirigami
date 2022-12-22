import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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


def filt_dset(dset, feats):
    out = []
    for row in dset:
        out_row = {}
        out_row["seq"] = row["seq"]
        out_row["dssr"] = row["dssr"]
        for feat in feats:
            out_row[feat] = row[feat]
        out.append(out_row)
    return out


feat_choices = {"pf", "pfold", "petfold", "pfile", "rnafold", "prob_pair",
                "centroid", "subopt0", "subopt1", "subopt2", "subopt3"} 
dist_choices = ("PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN")


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--norm", type=str, default="InstanceNorm2d", choices=["InstanceNorm2d", "BatchNorm2d"])
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--feats", type=str, choices=feat_choices, nargs="+", default=[])
    parser.add_argument("--acc-grad", type=int, default=32)
    parser.add_argument("--prec", type=str, default="32")
    parser.add_argument("--amp", type=str, default="native")
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--transfer-epochs", type=int, default=10000)
    parser.add_argument("--dists", type=str, choices=dist_choices, nargs="+", default=[])
    parser.add_argument("--bin-weight", type=float, default=0.0)
    parser.add_argument("--inv-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bins", type=str, default="torch.arange(2, 21, .5)")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.feats:
        args.feats = list(set(args.feats))

    if not args.sync:
        os.environ["WANDB_MODE"] = "offline"

    if args.prec.isdecimal():
        args.prec = int(args.prec) 

    bins = torch.arange(2, 21, .5)
    learner = KirigamiModule(n_blocks=32, 
                             n_channels=32,
                             p=args.dropout,
                             lr=args.lr,
                             arch="QRNA",
                             norm=args.norm,
                             transfer=False,
                             post_proc="greedy",
                             bins=bins,
                             pos_weight=args.pos_weight,
                             con_weight=(1 - args.bin_weight - args.inv_weight),
                             inv_weight=args.inv_weight,
                             bin_weight=args.bin_weight,
                             feats=args.feats,
                             dists=args.dists,
                             n_cutoffs=1000)
    
    # learner.dists = args.dists
    # learner.bins = args.bins
    # learner.hparams["bins"] = args.bins
    # learner._add_dists(args.dists, opt_types="both")

    
    data_path = Path(".")
    train_dataset = torch.load(data_path / "TR0-all.pt")
    val_dataset = torch.load(data_path / "VL1-all.pt")
    test_dataset = torch.load(data_path / "TS0-all.pt")

    train_dataset = filt_dset(train_dataset, args.feats)
    val_dataset = filt_dset(val_dataset, args.feats)
    test_dataset = filt_dset(test_dataset, args.feats)

    densify(train_dataset)
    densify(val_dataset)
    densify(test_dataset)

    if torch.cuda.device_count() > 1:
        densify(train_dataset)
        densify(val_dataset)
        densify(test_dataset)
        strategy = "ddp"
        devices = -1
    else:
        strategy = None
        devices = None

    wandb_logger = WandbLogger(project="kirigami", log_model=True)
    wandb_logger.experiment.log_code(".")

    mcc_checkpoint = ModelCheckpoint(monitor="pre/val/proc/mcc", mode="max", verbose=True)
    last_checkpoint = ModelCheckpoint(save_last=True, verbose=True)
    callbacks = [mcc_checkpoint, last_checkpoint]

    data_mod = DataModule(train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          feats=args.feats,
                          dists=args.dists,
                          batch_size=args.batch_size,
                          bins=bins)

    trainer = pl.Trainer(callbacks=callbacks,
                         max_time="01:23:00:00",
                         # max_epochs=args.epochs,
                         logger=wandb_logger,
                         # auto_lr_find=True,
                         accelerator="auto",
                         devices=devices,
                         strategy=strategy,
                         precision=args.prec,  
                         amp_backend=args.amp,
                         check_val_every_n_epoch=5,
                         # val_check_interval=.1,
                         accumulate_grad_batches=args.acc_grad)
    trainer.fit(learner, datamodule=data_mod)
    trainer.test(learner, ckpt_path="best", datamodule=data_mod)
    learner.load_from_checkpoint(checkpoint.best_model_path)


if __name__ == "__main__":
    main()

