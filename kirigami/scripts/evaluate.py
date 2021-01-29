import sys
import os
import json
import argparse
import pathlib
from multipledispatch import dispatch
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import munch
from kirigami.utils.data_utils import *
from kirigami.nn.SPOT import *
from kirigami.nn.Embedding import *
from kirigami.utils.utils import *


__all__ = ['evaluate']


@dispatch(argparse.Namespace)
def evaluate(args) -> None:
    config = path2munch(args.config)
    in_list = args.in_list
    quiet = args.quiet
    return evaluate(config, in_list, out_file, quiet)


@dispatch(munch.Munch, pathlib.Path, pathlib.Path, bool)
def evaluate(config, in_list, out_file, quiet) -> None:
    '''Evaluates model from config file'''
    if os.path.exists(config.training.best):
        saved = torch.load(config.training.best)
    else:
        saved = torch.load(config.training.checkpoint)

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    evaluate_set = BpseqDataset(in_list)
    evaluate_loader = DataLoader(evaluate_set)
    evaluate_loop = tqdm(evaluate_loader) if not quiet else evaluate_loader
    loss_func = getattr(nn, config.loss_func.class_name)(**config.loss_func.params)

    with open(in_list, 'r') as f:
        files = f.read().splitlines()
    names = []
    for file in files:
        basename = os.path.basename(file)
        basename, _ = os.path.splitext(file)
        names.append(basename)

    with open(out_file, 'w') as f:
        evaluate_loss = 0.
        for name, (seq, lab) in zip(names, evaluate_loop):
            pred = model(seq)
            loss = loss_func(pred, lab)
            evaluate_loss += loss
            F1, MCC = calcF1MCC( 
            f.write(f'{
    test_loss_mean = test_loss / len(test_loop)
    print(f'Mean test loss: {test_loss_mean}')
