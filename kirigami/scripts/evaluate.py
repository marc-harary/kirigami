import sys
import os
import json
import argparse
from multipledispatch import dispatch
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import munch
from utils.data_utils import *
from nn.SPOT import *
from nn.Embedding import *
from utils.data_utils import *

__all__ = ['evaluate']

@dispatch(argparse.Namespace)
def evaluate(args) -> None:
    config = path2munch(args.config)
    return evaluate(config)

@dispatch(munch.Munch)
def evaluate(config) -> None:
    '''Evaluates model from config file'''
    if os.path.exists(config.training.best):
        saved = torch.load(config.training.best)
    else:
        saved = torch.load(config.training.checkpoint)

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    dataset_class = TensorDataset if config.data.pre_embed else BpseqDataset
    test_set = dataset_class(config.data.test_list)
    test_loader = DataLoader(test_set, batch_size=config.data.batch_size)
    test_loop = tqdm(test_loader) if config.training.show_bar else test_loader
    loss_func = getattr(nn, config.loss_func.class_name)(**config.loss_func.params)

    test_loss = 0.
    for seq, lab in test_loop:
        pred = model(seq)
        loss = loss_func(pred, lab)
        test_loss += loss
    test_loss_mean = test_loss / len(test_loop)
    print(f'Mean test loss: {test_loss_mean}')
