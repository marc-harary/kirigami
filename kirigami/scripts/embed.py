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
from kirigami.utils.convert import bpseq2tensor, path2munch, st2tensor


@dispatch(Namespace)
def embed(args: Namespace) -> List[Path]:
    """Evaluates model from config file"""
    return embed(args.file_type, args.in_list,  args.out_directory, args.quiet) 


@dispatch(str, Path, Path, bool)
def embed(file_type: str,
          in_list: Path,
          out_dir: Path,
          quiet: bool = False) -> List[Path]:
    """Evaluates model from config file"""

    if file_type == "bpseq":
        embedding = bpseq2tensor
    elif file_type == "st":
        embedding = st2tensor
    else:
        raise ValueError("Unknown filetype")

    os.path.exists(out_dir) or os.mkdir(out_dir)

    with open(in_list, "r") as f:
        in_files = f.read().splitlines()

    out_files = []
    for file in in_files:
        out_file = os.path.basename(file)
        out_file, _ = os.path.splitext(out_file)
        out_file += ".pt"
        out_file = os.path.join(out_dir, out_file) 
        out_files.append(out_file)
        
    zipped = zip(in_files, out_files)
    loop = zipped if quiet else tqdm(zipped)

    for in_file, out_file in loop:
        with open(in_file, "r") as f:
            txt = f.read()
        try:
            pair = embedding(txt)
        except:
            print(in_file)
            import sys
            sys.exit(1)
        torch.save(pair, out_file)

    return out_files
