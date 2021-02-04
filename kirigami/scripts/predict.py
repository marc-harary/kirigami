import os
from argparse import Namespace
from pathlib import Path
from multipledispatch import dispatch
from munch import Munch
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from kirigami.utils.data import FastaDataset
from kirigami.utils.convert import binarize, tensor2bpseq
from kirigami.utils.path import *
from kirigami.nn.MainNet import MainNet


__all__ = ['predict']


@dispatch(Namespace)
def predict(args: Namespace) -> None:
    config = path2munch(args.config)
    return predict(config=config,
                   in_file=args.in_file,
                   out_file=args.out_file,
                   out_dir=args.out_directory,
                   quiet=args.quiet)


@dispatch(config: Munch, Path, Path, Path, bool)
def predict(config: Munch,
            in_file: Path,
            out_file: Path,
            out_dir: Path,
            quiet: bool = False) -> None:
    '''Evaluates model from config file'''
    try:
        saved = torch.load(config.data.best)
    except:
        saved = torch.load(config.data.checkpoint)
    else:
        raise FileNotFoundError('Can\'t find checkpoint files')

    os.path.exists(out_dir) or os.mkdir(out_dir)

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])

    out_bpseqs = []
    fp = open(out_file, 'w')
    with open(in_file, 'r') as f:
        files = f.read().splitlines()
    for file in files:
        out_bpseq = os.path.basename(file)
        out_bpseq = os.path.join(out_dir, out_bpseq)
        out_bpseqs.append(out_bpseq)
        fp.write(out_bpseqs+'\n')

    dataset = FastaDataset(in_file, quiet)
    dataloader = DataLoader(dataset)
    loop_zip = zip(out_bpseqs, dataloader)
    loop = loop_zip if quiet else tqdm(loop_zip)

    for out_bpseq, sequence in loop:
        pred = model(sequence)
        pred = binarize(pred)
        bpseq_str = tensor2bpseq(sequence, pred)
        with open(out_bpseq, 'w') as f:
            f.write(bpseq_str+'\n')
