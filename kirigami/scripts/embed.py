import argparse
import os
from glob import glob
from typing import Callable, List
from pathlib import Path

import torch
from tqdm import tqdm

from kirigami.utils.data import *
from kirigami.utils.convert import *


class Embed:
    """namespace for file embedding"""

    embed_fcn: Callable
    in_files: List[Path]
    out_file: Path
    tensor_dim: int
    concatenate: bool
    pad_length: int
    device: torch.device
    
    def __init__(self,
                embed_fcn: Callable,
                in_files: List[Path],
                out_file: Path,
                tensor_dim: int,
                concatenate: bool,
                pad_length: int,
                device: torch.device) -> None:
        self.embed_fcn = embed_fcn
        self.in_files = in_files
        self.out_file = out_file
        self.tensor_dim = tensor_dim
        self.concatenate = concatenate
        self.pad_length = pad_length
        self.device = device


    def run(self) -> None:
        N = len(self.in_files)
        seqs = torch.empty(N, 4, self.pad_length, dtype=torch.uint8) 
        labs =  torch.empty(N, 1, self.pad_length, self.pad_length, dtype=torch.uint8) 
        for i, file in tqdm(enumerate(self.in_files)):
            with open(file, "r") as f:
                txt = f.read()
            seq, lab = self.embed_fcn(txt, dim=self.tensor_dim, pad_length=self.pad_length, device=self.device)
            if self.concatenate:
                seq = concatenate_tensor(seq)
            seqs[i], labs[i] = seq, lab
        dset = torch.utils.data.TensorDataset(seqs, labs)
        torch.save(dset, self.out_file)


    @classmethod
    def from_namespace(cls, args: argparse.Namespace):
        if hasattr(args, "in_directory"):
            in_files = glob(str(args.in_directory / "*"))
        elif hasattr(args, "in_list"):
            with open(args.in_list, "r") as f:
                in_files = f.read().splitlines()
        if args.file_type == "bpseq":
            embed_fcn = bpseq2sparse if args.sparse else bpseq2dense
        elif args.file_type == "st":
            embed_fcn = st2sparse if args.sparse else st2dense
        return cls(embed_fcn=embed_fcn,
                   in_files=in_files,
                   out_file=args.out_file,
                   tensor_dim=args.tensor_dim,
                   concatenate=args.concatenate,
                   pad_length=args.pad_length,
                   device=torch.device(args.device))
