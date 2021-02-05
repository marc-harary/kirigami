'''predict `FASTA` files based on input config file'''

import os
from argparse import Namespace
from pathlib import Path
from typing import List

from multipledispatch import dispatch
from munch import Munch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from kirigami.utils.data import FastaDataset
from kirigami.utils.convert import binarize, tensor2bpseq, path2munch
from kirigami.nn.MainNet import MainNet


__all__ = ['predict']


@dispatch(Namespace)
def predict(args: Namespace) -> List[Path]:
    '''Evaluates model from config file'''
    config = path2munch(args.config)
    return predict(config, args.in_list, args.out_list, args.out_directory, args.quiet)


@dispatch(Munch, Path, Path, Path, bool)
def predict(config: Munch,
            in_list: Path,
            out_list: Path,
            out_dir: Path,
            quiet: bool = False) -> List[Path]:
    '''Evaluates model from config file'''
    try:
        saved = torch.load(config.data.best)
    except FileNotFoundError:
        saved = torch.load(config.data.checkpoint)
    else:
        raise FileNotFoundError('Can\'t find checkpoint files')

    os.path.exists(out_dir) or os.mkdir(out_dir)

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])

    bpseqs = []
    fp = open(out_list, 'w')
    with open(in_list, 'r') as f:
        fastas = f.read().splitlines()
    for fasta in fastas:
        bpseq = os.path.basename(fasta)
        bpseq = os.path.join(out_dir, bpseq)
        bpseq.append(bpseq)
        fp.write(bpseq+'\n')
    fp.close()

    dataset = FastaDataset(in_list, quiet)
    loader = DataLoader(dataset)
    loop_zip = zip(bpseqs, loader)
    loop = loop_zip if quiet else tqdm(loop_zip)

    for bpseq, sequence in loop:
        pred = model(sequence)
        pred = binarize(pred)
        bpseq_str = tensor2bpseq(sequence, pred)
        with open(bpseq, 'w') as f:
            f.write(bpseq_str+'\n')

    return bpseqs
