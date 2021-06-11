import os
import sys
import csv
from pathlib import Path
from argparse import Namespace
from typing import *

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import kirigami.nn
from kirigami._globals import *
from kirigami.utils.data import *
from kirigami.utils.process import *
from kirigami.utils.convert import *
from kirigami.utils.utils import *
from kirigami import binarize


__all__ = ["Test"] 


class Test:
    """namespace for test script"""

    out_file: Path
    model: nn.Module
    criterion: Callable
    test_loader: DataLoader
    test_collater: Callable
    thres_prob: float
    canonicalize: bool
    symmetrize: bool
    show_sequence_bar: bool
    
    def __init__(self,
                 out_file: Path,
                 model: nn.Module,
                 criterion: Callable,
                 test_loader: DataLoader,
                 test_collater: Callable,
                 thres_prob: float = 0.5,
                 canonicalize: bool = True,
                 symmetrize: bool = True,
                 show_sequence_bar: bool = True) -> None:
        self.out_file = out_file
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.test_collater = test_collater
        self.thres_prob = thres_prob
        self.canonicalize = canonicalize
        self.symmetrize = symmetrize
        self.show_sequence_bar = show_sequence_bar


    def run(self) -> None:
        fp = open(self.out_file, "w")
        writer = csv.writer(fp)
        writer.writerow(["length","raw_loss","bin_loss","tp","fp","tn","fn","mcc",
                         "f1","pred_pairs","ground_pairs"])

        loop = tqdm(self.test_loader) if self.show_sequence_bar else test_loader
        for seq, lab in loop:
            seq_coll, lab_coll = self.test_collater(seq, lab)
            seq_non_cat = seq_coll.squeeze()[:4,:,0]
            seq_str = dense2sequence(seq_non_cat)
            seq_length = int(seq_non_cat.sum().item())
            ground_pairs = int(lab_coll.sum().item() / 2)

            raw_pred = self.model(seq_coll)
            raw_loss = self.criterion(raw_pred, lab_coll.reshape_as(raw_pred))
            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                raw_pred = torch.nn.functional.sigmoid(raw_pred) 

            bin_pred = binarize(lab=raw_pred,
                                seq=seq_str,
                                min_dist=4,
                                thres_pairs=2**15,
                                thres_prob=self.thres_prob,
                                symmetrize=self.symmetrize,
                                canonicalize=self.canonicalize)

            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                bin_loss = torch.nn.functional.binary_cross_entropy(bin_pred, lab_coll.reshape_as(bin_pred))
            else:
                bin_loss = self.criterion(bin_pred, lab_coll)

            pred_contact_map = dense2pairmap(bin_pred, seq_length=seq_length)
            lab_contact_map = dense2pairmap(lab_coll, seq_length=seq_length)
            scores = get_scores(pred_contact_map, lab_contact_map)

            writer.writerow([seq_length,raw_loss.item(),bin_loss.item(),scores.tp,scores.fp,
                             scores.tn,scores.fn,scores.mcc,scores.f1,scores.pred_pairs,ground_pairs])

        fp.close()


    @classmethod
    def from_namespace(cls, args: Namespace):
        if args.model_device == "cuda" and torch.cuda.is_available():
            model_device = torch.device("cuda")
        else:
            model_device = torch.device("cpu")

        checkpoint = torch.load(args.checkpoint_file)
        criterion = eval(args.criterion)
        module_list = []
        for layer in args.layers:
            module_list.extend(flatten_module(eval(layer)))
        model = nn.Sequential(*module_list)
        model.to(model_device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_collater = lambda x: x
        if args.test_filetype == "bpseq-lst":
            test_set = BpseqDataset(list_file=args.test_file, device=data, batch_load=False)
        elif args.test_filetype == "pt-lst":
            test_set = EmbeddedDataset(list_file=args.test_file, device=data, batch_load=False)
        elif args.test_filetype == "st-lst":
            test_set = StDataset(list_file=args.test_file, device=data, batch_load=False)
        elif args.test_filetype == "pt":
            test_set = torch.load(args.test_file)
            test_coll_list = []
            seq, _ = test_set[0]
            if seq.device != model_device:
                test_coll_list.append(lambda x, y: (x.to(model_device), y.to(model_device)))
            if seq.is_sparse:
                test_coll_list.append(lambda x, y: (x.to_dense(), y.to_dense()))
            if seq.size(-1) != seq.size(-2): # sequence is concatenated
                test_coll_list.append(lambda x, y: (concatenate_batch(x), y))
            if seq.dtype != torch.float:
                test_coll_list.append(lambda x, y: (x.float(), y.float()))
            test_collater = compose_functions(reversed(test_coll_list))
        test_loader = DataLoader(test_set, shuffle=False, batch_size=1)


        return cls(out_file=args.out_file,
                   model=model,
                   criterion=criterion,
                   test_loader=test_loader,
                   test_collater=test_collater,
                   thres_prob=args.thres_prob,
                   canonicalize=not args.disable_canonicalize,
                   symmetrize=not args.disable_symmetrize,
                   show_sequence_bar=not args.disable_sequence_bar)
