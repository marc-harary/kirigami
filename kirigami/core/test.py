import os
from pathlib import Path
from argparse import Namespace
from typing import List, Callable

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import kirigami.nn
from kirigami._globals import *
from kirigami.utils.data import *
from kirigami.utils.convert import *


__all__ = ["Test"] 


class Test:
    """namespace for test script"""

    model: nn.Module
    criterion: Callable
    test_loader: DataLoader
    filenames: List[Path]
    out_file: Path
    canonicalize: bool
    symmetrize: bool
    thres_prob: float
    show_sequence_bar: bool
    
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 test_loader: DataLoader,
                 filenames: List[Path],
                 out_file: Path,
                 canonicalize: bool = True,
                 symmetrize: bool = True,
                 thres_prob: float = 0.5,
                 show_sequence_bar: bool = True):
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.filenames = filenames
        self.out_file = out_file
        self.canonicalize = canonicalize
        self.symmetrize = symmetrize
        self.show_sequence_bar = show_sequence_bar


    def run(self) -> None:
        zipped = zip(self.filenames, self.test_loader)
        loop = tqdm(zipped) if self.show_bar else zipped
        fp = open(self.out_file, "w")
        writer = csv.writer(fp)
        writer.writerow(["basename","loss","tp","fp","tn","fn","mcc","f1","ground_pairs","pred_pairs"])
        loss_tot, f1_tot, mcc_tot = 0., 0., 0.
        for filename, (seq, lab) in loop:
            pred = model(seq)
            loss = float(criterion(pred, lab))
            pred = binarize(pred, thres=thres)
            pair_map_pred, pair_map_lab = tensor2pairmap(pred), tensor2pairmap(lab)
            basename = os.path.basename(filename)
            basename, _ = os.path.splitext(basename)
            out = get_scores(pair_map_pred, pair_map_lab)
            f1_tot += out.f1
            mcc_tot += out.mcc
            loss_tot += loss
            writer.writerow([basename, loss, *list(out)])
            bpseq_str = tensor2bpseq(seq, pred)
            with open(out_bpseq, "w") as f:
                f.write(bpseq_str+"\n")
        fp.close()


    @classmethod
    def from_namespace(cls, args: Namespace):
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = nn.Sequential(*[eval(layer) for layer in args.layers])
        model.to(device)
        criterion = eval(args.criterion)

        if args.test_filetype == "bpseq-lst":
            test_set = BpseqDataset(list_file=args.test_file,
                                    device=data,
                                    batch_load=False)
        elif args.test_filetype == "pt-lst":
            test_set = EmbeddedDataset(list_file=args.test_file,
                                       device=data,
                                       batch_load=False)
        elif args.test_filetype == "st-lst":
            test_set = StDataset(list_file=args.test_file,
                                 device=data,
                                 batch_load=False)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

        with open(args.test_file, "w") as f:
            filenames = f.read().splitlines()
    
        return cls(model=model,
                   criterion=criterion,
                   test_loader=test_loader,
                   filenames=args.filenames,
                   out_file=args.out_file,
                   thres_prob=args.thres_prob,
                   canonicalize=not args.disable_canonicalize,
                   symmetrize=not args.disable_symmetrize,
                   show_sequence_bar=not args.disable_sequence_bar)
