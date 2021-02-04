import os
from pathlib import Path
from argparse import Namespace
import tempfile
from munch import Munch
from multipledispatch import dispatch
from tqdm import tqdm
import torch
from torch.nn import *
from torch.utils.data import DataLoader
from kirigami.utils.data import BpseqDataset
from kirigami.utils.utilities import *
from kirigami.utils.path import *


__all__ = ['evaluate']


@dispatch(Namespace)
def evaluate(args: Namespace) -> None:
    config = path2munch(args.config)
    return evaluate(config=config,
                    in_list=args.in_list,
                    out_list=args.out_list,
                    out_csv=args.out_csv,
                    out_dir=args.out_dir,
                    quiet=args.quiet)


@dispatch(Munch, Path, Path, Path, Path, bool)
def evaluate(config: Munch,
             in_list: Path,
             out_list: Path,
             out_csv: Path,
             out_dir: Path
             quiet: bool = False) -> None:
    '''Evaluates model from config file'''
    try:
        saved = torch.load(config.data.best)
    except:
        saved = torch.load(config.data.checkpoint)
    else:
        raise FileNotFoundError('Can\'t find checkpoint files')

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    with open(in_list, 'r') as f:
        in_bpseqs = f.read().splitlines()
    out_bpseqs = []
    for in_bpseq in in_bpseqs:
        out_bpseq = os.path.basename(in_bpseq)
        out_bpseq = os.path.join(out_dir, out_bpseq)
        out_bpseqs.append(out_bpseq)

    dataset = BpseqDataset(in_list, quiet)
    loader = DataLoader(dataset)
    loop_zip = zip(out_bpseqs, loader)
    loop = loop_zip if quiet else tqdm(loop_zip)

    loss_tot = 0.
    fp = open(out_file, 'w')
    writer = csv.writer(fp)
    writer.writerow(['basename', 'loss', 'mcc', 'f1'])
    for out_bpseq, (sequence, ground) in loop:
        pred = model(sequence)
        loss = loss_func(pred, ground)
        loss_tot += loss
        pred = binarize(pred)
        pair_map_pred, pair_map_ground = tensor2pairmap(pred), tensor2pairmap(ground)
        mcc, f1 = calcMCCF1(pair_map_pred, pair_map_ground)
        writer.writerow([basename, loss, mcc, f1])
    fp.close()

    if not quiet:
       mean_loss = loss_tot / len(loader)
       print(f'Mean loss for test set: {mean_loss}')
