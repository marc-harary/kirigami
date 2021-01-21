import sys
import os
import json
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from munch import munchify
from kirigami.utils.data_utils import *
from kirigami.nn.SPOT import *
from kirigami.nn.Embedding import *
from kirigami.utils.data_utils import *


def evaluate(config: Munch) -> None:
    if os.path.exists(conf.training.best):
        saved = torch.load(conf.training.best)  
    else:
        saved = torch.load(conf.training.checkpoint)

    model = MainNet(conf.model)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    dataset_class = TensorDataset if conf.data.pre_embed else BpseqDataset
    test_set = dataset_class(conf.data.test_list)
    test_loader = DataLoader(test_set, batch_size=conf.data.batch_size)
    test_loop = tqdm(test_loader) if conf.training.show_bar else test_loader
    loss_func = getattr(nn, conf.loss_func.class_name)(**conf.loss_func.params)

    test_loss = 0.
    for seq, lab in test_loop:
        pred = model(seq)
        loss = loss_func(pred, lab)
        test_loss += loss
    test_loss_mean = test_loss / len(test_loop)
    print(f'Mean test loss: {test_loss_mean}')
