import os
from pathlib import Path
from argparse import Namespace
from typing import List

from munch import Munch
from multipledispatch import dispatch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from kirigami.utils.data import BpseqDataset
from kirigami.utils.convert import bpseq2tensor
from kirigami.nn.MainNet import MainNet


@dispatch(Namespace)
def embed(args: Namespace) -> List[Path]:
    '''Evaluates model from config file'''
    config = path2munch(args.config)
    return embed(args.in_list, args.out_directory, quiet) 


@dispatch(Munch, Path, Path, float, bool, bool)
def embed(config: Munch,
          out_dir: Path,
          quiet: bool = False) -> List[Path]:
    with open(config.in_list, 'r') as f:
        in_files = f.read().splitlines()

    out_files = []
    for file in in_files:
        out_file = os.path.basename(file)
        out_file, _ = os.path.splitext(out_file)
        out_file += '.pt'
        out_file = os.path.join(out_dir, out_file) 
        out_files.append(out_file)
        
    zipped = zip(in_files, out_files)
    loop = zipped if quiet else tqdm(zipped)

    for in_file, out_file in loop:
        with open(in_file, 'r') as f:
            bpseq_str = f.read()
        pair = bpseq2tensor(bpseq_str)
        torch.save(pair, out_file)

    return out_files
