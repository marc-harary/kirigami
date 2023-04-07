# Kirigami

RNA secondary structure prediction via deep learning.

From [Wikipedia](https://en.wikipedia.org/wiki/Kirigami):

> Kirigami (切り紙) is a variation of origami, the Japanese art of folding paper. In kirigami, the paper is cut as well as being folded, resulting in a three-dimensional design that stands away from the page.

The Kirigami pipeline both folds RNA molecules via a fully convolutional neural network (FCN) and uses Nussinov-style dynamic programming to recursively cut them into subsequences for pre- and post-processing.

## Installation
No specific setup is necessary; the small number of packages required are in listed `requirements.txt`. For example, one might run:
```bash
$ pip -m venv kirigami-venv
$ source kirigami-venv/bin/activate
$ pip install -r requirements.txt
$ python run.py --help
```

## Overview

For ease of use and reproducibility, all scripts are written idiomatically according to the [Lightning](https://www.pytorchlightning.ai) specification for PyTorch with as little application-specific code as possible. The three principal classes compromising the module are:

1. `kirigami.layers.ResNet`: A standard `torch.nn.Module` comprising the main model;
2. `kirigami.data.DataModule`: Used for downloading, embedding, pickling, and loading samples;
3. `kirigami.data.KirigamiModule`: Wraps `kirigami.layers.ResNet` and includes a small number of hooks for reproducible logging, checkpointing, loops for training, etc. Please see the [documentation](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) for the main API.

```bash
kirigami
├── __init__.py
├── data.py
├── layers.py
├── learner.py
└── utils.py
```

## Usage

The primary entrypoint for Kirigami is the [LightningCLI](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html), used as follows:

```bash
$ python run.py --help
usage: run.py [-h] [-c CONFIG] [--print_config[=flags]] {fit,validate,test,predict} ...

pytorch-lightning trainer command line tool

options:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or
                        more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  {fit,validate,test,predict}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set.
    predict             Run inference on your data.
```

## Model architecture and hyperparameters

Kirigami consists of an extremely simple residual neural network (RNN) architecture that can be found in `kirigami/layers.py`, with primary network API being `kirigami.layers.ResNet`. The hyperparameters for the model are as follows:

```bash
<class 'kirigami.learner.KirigamiModule'>:
  --model.n_blocks N_BLOCKS
  		(required, type: int)
  --model.n_channels N_CHANNELS
  		(required, type: int)
  --model.kernel_sizes [ITEM,...]
  		(required, type: Tuple[int, int])
  --model.dilations [ITEM,...]
  		(required, type: Tuple[int, int])
  --model.activation ACTIVATION
  		(required, type: str)
  --model.dropout DROPOUT
```

These are:
1. `n_blocks`: The total number of residual neural network blocks (i.e., `kirigami.layers.ResNetBlock`);
2. `n_channels`: The number of hidden channels for each block;
3. `kernel_sizes`: The dimensions of the kernels for the first and second `torch.nn.Conv2D` layers in each block;
4. `dilations`: The dilations for said convolutional layers;
5. `activation`: The class name for the [non-linearities](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearit ) in each block;
6. `dropout`: The dropout probability for the `torch.nn.Dropout` layer in each block.

## Data

All data will be automatically downloaded from [Dropbox](https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip) by the `DataModule.setup` and `DataModule.prepare_data` methods in `kirigami/data.py`. Please see the documentation for the [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) API for more detail. No parameters are required for the use of the `DataModule` class aside from the directory into the files will be pickled:

```bash
<class 'kirigami.data.DataModule'>:
  --data CONFIG         Path to a configuration file.
  --data.data_dir DATA_DIR
                        (type: Any, default: ./)
```

Data used for training, validation, and testing are taken from the bpRNA database in the form of TR0, VL0, and TS0, respectively, as used by [SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA), [MXfold2](https://github.com/mxfold/mxfold2), and [UFold](https://github.com/uci-cbcl/UFold). The `.st` files in the URL above were uploaded by the authors of SPOT-RNA.
