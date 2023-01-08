import sys
import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb

from kirigami.data import DataModule
from kirigami.loss import ForkLoss
from kirigami.learner import KirigamiModule


feat_choices = {"pf", "pfold", "petfold", "pfile", "rnafold", "prob_pair",
                "centroid", "subopt0", "subopt1", "subopt2", "subopt3"} 
dist_choices = ("PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN")


def main():
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sync", action="store_true")

    parser.add_argument("--feats", type=str, choices=feat_choices, nargs="+", default=[])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-blocks", type=int, default=32)
    parser.add_argument("--n-channels", type=int, default=32)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1])
    parser.add_argument("--activation", type=str, default="ReLU")

    parser.add_argument("--pos-weight", type=float, default=0.5)
    parser.add_argument("--con-weight", type=float, default=1.)

    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--transfer-epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--acc-grad", type=int, default=1)
    parser.add_argument("--prec", type=str, default="32")
    parser.add_argument("--amp", type=str, default="native")
    
    args = parser.parse_args()

    if args.feats:
        args.feats = list(set(args.feats))

    if not args.sync:
        os.environ["WANDB_MODE"] = "offline"

    if args.prec.isdecimal():
        args.prec = int(args.prec) 

    model_kwargs = {}
    for kwarg_name in ["n_blocks", "n_channels", "kernel_sizes", "dropout",
                       "dilations", "activation"]:
        model_kwargs[kwarg_name] = getattr(args, kwarg_name)
    
    crit_kwargs = dict(pos_weight=args.pos_weight, con_weight=args.con_weight)
        
    bins = torch.arange(2, 21, .5)
    learner = KirigamiModule(model_kwargs=model_kwargs,
                             crit_kwargs=crit_kwargs,
                             lr=args.lr,
                             optim=args.optim,
                             transfer=False,
                             n_cutoffs=1000)

    data_path = Path("/gpfs/ysm/project/pyle/mah258/data-all")
    data_mod = DataModule(train_path=data_path / "TR0-all.pt",
                          val_path=data_path / "VL0-all.pt",
                          test_path=data_path / "TS1-all.pt", 
                          densify=True,
                          batch_size=args.batch_size,
                          dists=[],
                          feats=args.feats)

    wandb_logger = WandbLogger(project="kirigami", log_model=True)
    wandb_logger.experiment.log_code(".")

    mcc_checkpoint = ModelCheckpoint(monitor="pre/val/proc/mcc", mode="max", verbose=True)
    last_checkpoint = ModelCheckpoint(save_last=True, verbose=True)
    every_checkpoint = ModelCheckpoint(every_n_epochs=2, verbose=True, save_top_k=-1)
    callbacks = [mcc_checkpoint, last_checkpoint, every_checkpoint]
    # callbacks = [every_checkpoint]

    trainer = pl.Trainer(callbacks=callbacks,
                         max_time="01:23:00:00",
                         # max_epochs=args.epochs,
                         logger=wandb_logger,
                         # auto_lr_find=True,
                         accelerator="auto",
                         devices=None,
                         # strategy=strategy,
                         precision=args.prec,  
                         amp_backend=args.amp,
                         check_val_every_n_epoch=5,
                         # val_check_interval=.1,
                         accumulate_grad_batches=args.acc_grad)
    trainer.fit(learner, datamodule=data_mod)
    trainer.test(learner, ckpt_path="best", datamodule=data_mod)


if __name__ == "__main__":
    main()

