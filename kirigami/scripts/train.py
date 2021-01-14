import argparse
import json
from munch import munchify
import torch
from torch import nn
from ..nn.MainNet import *
from ..nn.Loss import *
from ..nn.Embedding import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', help='quiet', type=bool, default=False)
    parser.add_argument('--conf', help='path to configuration file', type=str)
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf_str = f.read()
        conf_dict = json.loads(conf_str)
        conf = munchify(conf_dict)

    with open(conf.data.train_list, 'r') as train_pointer:
        train_files = train_pointer.readlines()
        train_seqs = []
        for file in train_files:
            with open(file, 'r') as file_pointer:
                train_seqs.append(file_pointer.read())

    with open(conf.data.val_list, 'r') as val_pointer:
        val_files = val_pointer.readlines()
        val_seqs = []
        for file in val_files:
            with open(file, 'r') as val_pointer:
                val_seqs.append(val_pointer.read())


    net = MainNet(conf.layers)
    loss = Loss(conf.loss)

    for epoch in range(conf.train.epochs):
      pass  

    

if __name__ == '__main__':
    main()
