import sys
import os
from pathlib import Path
import math

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging

from argparse import ArgumentParser

import wandb

from kirigami.data import DataModule
from kirigami.loss import ForkLoss
from kirigami.learner import KirigamiModule
from kirigami.resnet import ResNetParallel


feat_choices = {"pf", "pfold", "petfold", "pfile", "rnafold", "prob_pair",
                "centroid", "subopt0", "subopt1", "subopt2", "subopt3"} 
dist_choices = ("PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN")


def main():
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sync", action="store_true")

    parser.add_argument("--acc-grad", type=int, default=1)
    parser.add_argument("--prec", type=str, default="32")
    parser.add_argument("--amp", type=str, default="native")

    parser.add_argument("--n-trunk-blocks", type=int, default=32)
    parser.add_argument("--n-trunk-channels", type=int, default=32)

    parser.add_argument("--n-con-blocks", type=int, default=32)
    parser.add_argument("--n-con-channels", type=int, default=32)

    parser.add_argument("--n-dist-blocks", type=int, default=32)
    parser.add_argument("--n-dist-channels", type=int, default=32)

    parser.add_argument("--feats", type=str, choices=feat_choices, nargs="+", default=[])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1])
    parser.add_argument("--activation", type=str, default="ReLU")

    parser.add_argument("--bin-step", type=float, default=1)
    parser.add_argument("--bin-min", type=float, default=2.)
    parser.add_argument("--bin-max", type=float, default=22.)
    parser.add_argument("--dist-types", type=str, choices=dist_choices, nargs="+", default=[])

    parser.add_argument("--con-weight", type=float, default=1.)
    parser.add_argument("--dist-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=0.5)

    parser.add_argument("--post-proc", type=str, default="greedy")

    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--batch-size", type=int, default=1)
    
    args = parser.parse_args()
    if args.feats:
        args.feats = list(set(args.feats))
    if not args.sync:
        os.environ["WANDB_MODE"] = "offline"
    if args.prec.isdecimal():
        args.prec = int(args.prec) 

    model_kwargs = {}
    for kwarg_name in ["n_trunk_blocks", "n_trunk_channels", "n_con_blocks",
        "n_con_channels", "n_dist_blocks", "n_dist_channels", "kernel_sizes",
        "dropout", "activation", "dist_types", "dilations"]:
        model_kwargs[kwarg_name] = getattr(args, kwarg_name)
    idx_min = math.floor(args.bin_min / args.bin_step + .5)
    idx_max = math.floor(args.bin_max / args.bin_step + .5)
    n_bins = idx_max - idx_min + 1
    model_kwargs["n_bins"] = n_bins

    crit_kwargs = {}
    for kwarg_name in ["pos_weight", "con_weight"]:
        crit_kwargs[kwarg_name] = getattr(args, kwarg_name)
    learner = KirigamiModule(model_kwargs=model_kwargs,
                             crit_kwargs=crit_kwargs,
                             post_proc=args.post_proc,
                             lr=args.lr,
                             dists=args.dist_types,
                             bin_min=args.bin_min,
                             bin_max=args.bin_max,
                             bin_step=args.bin_step,
                             optim=args.optim,
                             momentum=args.momentum,
                             n_cutoffs=1000,
                             transfer=True)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        learner.load_state_dict(ckpt["state_dict"], strict=False)


    data_path = Path("/gpfs/ysm/project/pyle/mah258/data-all")
    data_mod = DataModule(train_path=data_path / "TR1-all.pt",
                          val_path=data_path / "VL1-all.pt",
                          test_path=data_path / "TS1-all.pt", 
                          densify=False,
                          bin_step=args.bin_step,
                          bin_min=args.bin_min,
                          bin_max=args.bin_max,
                          batch_size=args.batch_size,
                          dists=args.dist_types,
                          feats=args.feats)

    wandb_logger = WandbLogger(project="kirigami", log_model=True)
    wandb_logger.experiment.log_code(".")

    mcc_checkpoint = ModelCheckpoint(monitor="transfer/val/proc/mcc", mode="max",
                                     verbose=True)
    last_checkpoint = ModelCheckpoint(save_last=True, verbose=True)
    swa = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [mcc_checkpoint, last_checkpoint]#, swa]

    trainer = pl.Trainer(callbacks=callbacks,
                         max_time="01:23:00:00",
                         max_epochs=args.max_epochs,
                         logger=wandb_logger,
                         # auto_lr_find=True,
                         accelerator="auto",
                         devices=None,
                         # strategy=strategy,
                         precision=args.prec,  
                         amp_backend=args.amp,
                         check_val_every_n_epoch=10,
                         # val_check_interval=.1,
                         accumulate_grad_batches=args.acc_grad)

    trainer.fit(learner, datamodule=data_mod)
    trainer.test(learner, ckpt_path="best", datamodule=data_mod)
    learner.load_from_checkpoint(checkpoint.best_model_path)


if __name__ == "__main__":
    main()

