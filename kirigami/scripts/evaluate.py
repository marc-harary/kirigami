'''Evaluates model on list of input files'''


import os
from pathlib import Path
from argparse import Namespace
from typing import List
import csv

from munch import Munch
from multipledispatch import dispatch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from kirigami.utils.data import BpseqDataset
from kirigami.utils.convert import path2munch, binarize, tensor2pairmap, calcf1mcc, tensor2bpseq
from kirigami.nn.MainNet import MainNet


__all__ = ['evaluate']


@dispatch(Namespace)
def evaluate(args: Namespace) -> List[Path]:
    '''Evaluates model from config file'''
    config = path2munch(args.config)
    return evaluate(config, args.in_list, args.out_directory, args.quiet)


@dispatch(Munch, Path, Path, bool)
def evaluate(config: Munch,
             in_list: Path,
             out_dir: Path,
             quiet: bool = False) -> List[Path]:
    '''Evaluates model from config file'''
    try:
        saved = torch.load(config.data.best)
    except FileNotFoundError:
        saved = torch.load(config.data.checkpoint)
    else:
        raise FileNotFoundError('Can\'t find checkpoint files')

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    os.path.exists(out_dir) or os.mkdir(out_dir)
    bpseq_dir = os.path.join(out_dir, 'bpseqs')
    os.path.exists(bpseq_dir) or os.mkdir(bpseq_dir)
    out_csv = os.path.join(out_dir, 'scores.csv')

    with open(in_list, 'r') as f:
        in_bpseqs = f.read().splitlines()
    out_bpseqs = []
    for in_bpseq in in_bpseqs:
        out_bpseq = os.path.basename(in_bpseq)
        out_bpseq = os.path.join(bpseq_dir, out_bpseq)
        out_bpseqs.append(out_bpseq)

    dataset = BpseqDataset(in_list, quiet)
    loader = DataLoader(dataset)
    loop_zip = zip(out_bpseqs, loader)
    loop = loop_zip if quiet else tqdm(loop_zip)
    loss_func = eval(config.loss_func.class_name)(**config.loss_func.params)

    loss_tot = 0.
    fp = open(out_csv, 'w')
    writer = csv.writer(fp)
    writer.writerow(['basename', 'loss', 'mcc', 'f1'])
    for out_bpseq, (sequence, ground) in loop:
        pred = model(sequence)
        loss = float(loss_func(pred, ground))
        loss_tot += loss
        pred = binarize(pred)
        pair_map_pred, pair_map_ground = tensor2pairmap(pred), tensor2pairmap(ground)
        f1, mcc = calcf1mcc(pair_map_pred, pair_map_ground)
        bpseq_str = tensor2bpseq(sequence, pred)
        with open(out_bpseq, 'w') as f:
            f.write(bpseq_str+'\n')
        basename = os.path.basename(out_bpseq)
        basename, _ = os.path.splitext(basename)
        writer.writerow([basename, loss, mcc, f1])
    fp.close()

    if not quiet:
        mean_loss = loss_tot / len(loader)
        print(f'Mean loss for test set: {mean_loss}')

    return out_bpseqs
