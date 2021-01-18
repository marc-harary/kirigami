import os
import json
from tqdm import tqdm
from munch import munchify
import torch
from torch import nn
from torch.utils.data import DataLoader
from kirigami.nn.MainNet import *
from kirigami.nn.Embedding import *
from kirigami.nn.SPOT import *
from kirigami.utils.data_utils import *


def train(conf_path: str):
    with open(conf_path, 'r') as f:
        conf_str = f.read()
        conf_dict = json.loads(conf_str)
        conf = munchify(conf_dict)

    start_epoch = 0
    model = MainNet(conf.model)
    loss_func = getattr(nn, conf.loss_func.class_name)(**conf.loss_func.params)
    optimizer = getattr(torch.optim, conf.optim.class_name)(model.parameters(),
                                                            **conf.optim.params)

    train_set = BpseqDataset(conf.data.training_list)
    train_loader = DataLoader(train_set,
                              batch_size=conf.data.batch_size,
                              shuffle=conf.data.shuffle)
    if conf.data.validation_list:
        val_set = BpseqDataset(conf.data.validation_list)
        val_loader = DataLoader(val_set,
                                batch_size=conf.data.batch_size,
                                shuffle=conf.data.shuffle)

    if conf.resume:
        assert os.path.exists(conf.training.checkpoint), "Cannot find checkpoint file"
        checkpoint = torch.load(conf.training.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(start_epoch, conf.training.epochs):
        for seq, lab in tqdm(train_loader):
            pred = model(seq)
            loss = loss_func(pred, lab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   conf.checkpoint)
        print(f'Loss for epoch {epoch}: {loss}\n')

