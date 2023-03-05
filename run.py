# import sys
import os
# from pathlib import Path
# import math

# import torch
# import torch.nn as nn
# 
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins.environments import SLURMEnvironment
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
# 
from pytorch_lightning.cli import LightningCLI
# 
# from argparse import ArgumentParser
# 
import wandb

from kirigami.data import DataModule
from kirigami.loss import ForkLoss
from kirigami.learner import KirigamiModule
from kirigami.resnet import ResNetParallel


# feat_choices = {"pf", "pfold", "petfold", "pfile", "rnafold", "prob_pair",
#                 "centroid", "subopt0", "subopt1", "subopt2", "subopt3"} 
# dist_choices = ("PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN")


def main():
    os.environ["WANDB_MODE"] = "offline"
    cli = LightningCLI(KirigamiModule, DataModule)

if __name__ == "__main__":
    main()

