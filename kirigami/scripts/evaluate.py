import os
from argparse import Namespace
from tqdm import tqdm
import torch
from torch.nn import *
from torch.utils.data import DataLoader
from kirigami.utils.data import BpseqDataset
from kirigami.utils.utilities import *


__all__ = ['evaluate']


def evaluate(args: Namespace) -> None:
    '''Evaluates model from config file'''
    config = path2munch(args.config)

    try:
        saved = torch.load(config.data.best)
    except:
        saved = torch.load(config.data.checkpoint)
    else:
        raise FileNotFoundError('Can\'t find checkpoint files')

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])

    out_files = []
    with open(config.in_file, 'r') as f:
        in_files = f.read().splitlines()
        for file in in_files:
            basename = file = os.path.basename(file)
            file, _ = os.path.splitext(file)
            file += '.bpseq'
            file = os.path.join(args.out_directory, file)
            out_files.append((basename, file))

    dataset = BpseqDataset(args.in_file, args.quiet)
    loader = DataLoader(dataset)
    loop_zip = zip(out_files, loader)
    loop = loop_zip if args.quiet else tqdm(loop_zip)

    csv_list = ['basename,loss,mcc,f1\n']
    loss_tot = 0.
    for (basename, out_file), (sequence, label) in loop:
        pred = model(sequence)
        loss = loss_func(pred, label)
        loss_tot += loss
        pred = binarize(pred)
        pair_map_pred, pair_map_ground = tensor2pairmap(pred), tensor2pairmap(label)
        mcc, f1 = calcMCCF1(pair_map_pred, pair_map_ground)
        csv_list.append(f'{basename},{loss},{mcc},{f1}\n')

    if not args.quiet:
       mean_loss = loss_tot / len(loader)
       print(f'Mean loss for test set: {mean_loss}')

    with open(args.out_file, 'w') as f:
        for line in csv_list:
            csv_list.write(line)
