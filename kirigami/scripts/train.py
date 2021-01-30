import os
import json
import argparse
from multipledispatch import dispatch
from tqdm import tqdm
import munch
import torch
from torch import nn
from torch.utils.data import DataLoader
from kirigami.utils.utils import path2munch
from kirigami.utils.data_utils import *
from kirigami.nn.SPOT import *
from kirigami.nn.Embedding import *
from kirigami.nn.MainNet import *


__all__ = ['train']


@dispatch(argparse.Namespace)
def train(args) -> None:
    config = path2munch(args.config)
    return train(config, args.quiet)


@dispatch(munch.Munch, bool)
def train(config, quiet) -> None:
    '''Train deep network based on config files'''
    start_epoch = 0
    model = MainNet(config.model)
    loss_func = getattr(nn, config.loss_func.class_name)(**config.loss_func.params)
    optimizer = getattr(torch.optim, config.optim.class_name)(model.parameters(),
                                                            **config.optim.params)

    if not os.listdir(config.data.embedding_directory):
        train_bpseq_dataset = BpseqDataset(config.data.training_list)
        train_bpseq_dataset.embed(config.data.embedding_directory, config.data.embedding_list)

    train_set = TensorDataset(config.data.embedding_list):

    best_val_loss = float('inf')
    dataset_class = TensorDataset if config.data.pre_embed else BpseqDataset
    train_set = dataset_class(config.data.training_list)
    train_loader = DataLoader(train_set,
                              batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle)
    train_loop = tqdm(train_loader) if not quiet else train_loader


    if config.data.validation_list:
        val_set = dataset_class(config.data.validation_list)
        val_loader = DataLoader(val_set,
                                batch_size=config.data.batch_size,
                                shuffle=config.data.shuffle)
        val_loop = tqdm(val_loader) if not quiet else val_loader


    if config.resume:
        assert os.path.exists(config.training.best), "Cannot find checkpoint file"
        best = torch.load(config.training.best)
        model.load_state_dict(best['model_state_dict'])
        optimizer.load_state_dict(best['optimizer_state_dict'])
        start_epoch = best['epoch']
        loss = best['loss']
        best_val_loss = best['best_val_loss']


    for epoch in range(start_epoch, config.training.epochs):
        train_loss_tot = 0.
        for seq, lab in train_loop:
            pred = model(seq)
            loss = loss_func(pred, lab)
            train_loss_tot += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_mean = train_loss_tot / len(train_loader)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_mean},
                   config.training.checkpoint)

        if config.data.validation_list:
            val_loss_tot = 0.
            for seq, lab in val_loop:
                pred = model(seq)
                loss = loss_func(pred, lab)
                val_loss_tot += loss
                val_loss_mean = val_loss_tot / len(val_loader)
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_val_loss},
                           config.training.best)

        if epoch % config.training.print_frequency == 0:
            print(f'Mean training loss for epoch {epoch}: {train_loss_mean}')
            if config.data.validation_list:
                print(f'Mean validation loss for epoch {epoch}: {val_loss_mean}')
            print()
