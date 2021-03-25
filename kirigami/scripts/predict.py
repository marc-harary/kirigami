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


__all__ = ["predict"]


@dispatch(Namespace)
def predict(args: Namespace) -> List[Path]:
    """Evaluates model from config file"""
    config = path2munch(args.config)
    return predict(config=config,
                   in_list=args.in_list,
                   out_list=args.out_list,
                   out_dir=args.out_directory,
                   quiet=args.quiet,
                   disable_gpu=args.disable_gpu) 


@dispatch(Munch, Path, Path, Path, bool, bool)
def predict(config: Munch,
            in_list: Path,
            out_list: Path,
            out_dir: Path,
            quiet: bool = False,
            disable_gpu: bool = False) -> List[Path]:
    """Evaluates model from config file"""

    try:
        saved = torch.load(config.data.best)
    except FileNotFoundError:
        saved = torch.load(config.data.checkpoint)

    os.path.exists(out_dir) or os.mkdir(out_dir)

    device = DEVICE if not disable_gpu else torch.device("cpu")
    model = torch.nn.DataParallel(MainNet(config.model))
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()

    bpseqs = []
    fp = open(out_list, "w")
    with open(in_list, "r") as f:
        fastas = f.read().splitlines()
    for fasta in fastas:
        bpseq = os.path.basename(fasta)
        bpseq = os.path.join(out_dir, bpseq)
        bpseq.append(bpseq)
        fp.write(bpseq+"\n")
    fp.close()

    dataset = FastaDataset(in_list, quiet, device)
    loader = DataLoader(dataset)
    loop_zip = zip(bpseqs, loader)
    loop = loop_zip if quiet else tqdm(loop_zip)

    for bpseq, sequence in loop:
        pred = model(sequence)
        pred = binarize(pred)
        bpseq_str = tensor2bpseq(sequence, pred)
        with open(bpseq, "w") as f:
            f.write(bpseq_str+"\n")

    return bpseqs
