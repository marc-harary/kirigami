import os
import json
import argparse
from tqdm import tqdm
import munch
import torch
from torch.optim import *
from torch.nn import *
from torch.utils.data import DataLoader
from kirigami.utils.utilities import path2munch
from kirigami.utils.data import BpseqDataset
from kirigami.nn.MainNet import MainNet


__all__ = ['train']


def train(args: argparse.Namespace) -> None:
    '''Train deep network based on config files'''
    config = path2munch(args.config)

    start_epoch = 0
    model = MainNet(config.model)
    loss_func = eval(config.loss_func.class_name)(**config.loss_func.params)
    optimizer = eval(config.optim.class_name)(model.parameters(), **config.optim.params)

    best_val_loss = float('inf')
    train_set = BpseqDataset(config.data.training_list, args.quiet)
    train_loader = DataLoader(train_set,
                              batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle)
    train_loop = train_loader if args.quiet else tqdm(train_loader)

    if config.data.validation_list:
        val_set = BpseqDataset(config.data.validation_list, args.quiet)
        val_loader = DataLoader(val_set,
                                batch_size=config.data.batch_size,
                                shuffle=config.data.shuffle)
        val_loop = val_loader if args.quiet else tqdm(val_loader)

    if args.resume:
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
