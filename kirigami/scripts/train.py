import argparse
import os
import json
from munch import munchify
import torch
from torch import nn
from torch.utils.data import *
from ..nn.MainNet import *
from ..nn.Embedding import *
from ..utils.data_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', help='path to configuration file', type=str)
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf_str = f.read()
        conf_dict = json.loads(conf_str)
        conf = munchify(conf_dict)

    model = MainNet(conf.model)
    loss_func = getattr(nn, conf.loss_func.class)(**conf.loss_func.params)
    optimizer = getattr(nn, conf.optim.class)(model.parameters(),
                                          **conf.optim.params)

    train_set = BpseqDataset(conf.train_list)
    val_set = BpseqDataset(conf.validation_list)
    train_loader = DataLoader(train_set,
                              batch_size=conf.training.batch_size,
                              shuffle=conf.training.shuffle)
    val_loader = DataLoader(val_set,
                            batch_size=conf.training.batch_size,
                            shuffle=conf.training.shuffle)

    if conf.resume and os.path.exists(conf.training.checkpoint):
        checkpoint = torch.load(conf.training.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(conf.train.epochs):
        for seq, label in train_loader:
            pred = model(seq)
            loss = loss_func(pred, label)
            loss.backward()
            optimzer.step()
            optimizer.zero_grad()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   conf.checkpoint)
        print(f'Loss for epoch {epoch}: {loss}\n')


if __name__ == '__main__':
    main()
