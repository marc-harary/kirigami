import os
from pytorch_lightning.cli import LightningCLI
# from kirigami.learner import KirigamiModule
import kirigami
from kirigami.data import DataModule
# from kirigami.loss import ForkLoss
# from kirigami.resnet import ResNetParallel

def main():
    cli = LightningCLI(kirigami.KirigamiModule, kirigami.DataModule)

if __name__ == "__main__":
    main()

