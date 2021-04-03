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
from kirigami.utils.data import BpseqDataset, EmbeddedDataset, StDataset
from kirigami.utils.convert import *


__all__ = ["Test"] 


class Test:
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 test_loader: DataLoader,
                 filenames: List[Path],
                 out_file: Path,
                 thres: float = 0.5,
                 show_bar: bool = True):
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.filenames = filenames
        self.out_file = out_file
        self.show_bar = show_bar


    def run() -> None:
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
        if args.device == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                raise ValueError("CUDA is not available")
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
        else:
            raise ValueError("Invalid file type (pickled files cannot be tested; file names must be provided")
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

        with open(args.test_file, "w") as f:
            filenames = f.read().splitlines()
    
        return cls(model=model,
                   criterion=criterion,
                   test_loader=test_loader,
                   filenames=filenames,
                   out_file=args.out_file,
                   show_bar=args.show_bar)
