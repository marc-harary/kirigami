import os
import time
import logging
import datetime
from typing import Union, Callable
from pathlib import Path

import argparse
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kirigami._globals import *
from kirigami.utils.data import BpseqDataset, EmbeddedDataset, StDataset


__all__ = ["Train"] 


class Train:
    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion: Callable,
                 training_set: DataLoader,
                 validation_set: Union[DataLoader,None],
                 log_file: Union[Path,None],
                 training_checkpoint_file: Union[Path,None],
                 validation_checkpoint_file: Union[Path,None],
                 batch_size: int = 1,
                 shuffle: bool = True,
                 show_bar: bool = True,
                 disable_cuda: bool = False,
                 quiet: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.training_loader = DataLoader(training_set, shuffle=shuffle, batch_size=batch_size)
        if validation_set:
            self.validation_loader = DataLoader(validation_set, shuffle=shuffle, batch_size=batch_size)

        self.log_file = log_file
        self.checkpoint_path = checkpoint_file
        self.validation_file = validation_file
        self.disable_cuda = disable_cuda
        self.quiet = quiet

        self.device = device
        self.model.to(device)


    def log(self, message: str):
        if self.quiet:
            return
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO)
        logging.info(message)


    def run(self, resume: bool = False):
        self.log("Starting at " + str(datetime.datetime.now()))
            
        model.train()
        start_epoch = 0
        best_val_loss = float("inf")

        if resume:
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            self.log(f"Resuming at epoch {epoch} with loss {loss}")

        range_iterator = range(start_epoch, self.epochs)
        outer_loop = tqdm(range_iterator) if self.show_bar else range_iterator

        for epoch in outer_loop:
            start = datetime.datetime.now()
            self.log(f"Starting epoch {epoch} at {start}")

            train_loss_tot = 0.
            inner_loop = tqdm(self.training_loader) if self.show_bar else self.training_loader
            for seq, lab in inner_loop:
                pred = self.model(seq)
                loss = self.criterion(pred, lab)
                train_loss_tot += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_loss_mean = train_loss_tot / len(self.training_loader)
            state_dict = self.model.state_dict()
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss_mean},
                        self.checkpoint_file)

            if self.validation_loader:
                self.validate()

            end = datetime.datetime.now()
            delta = end - start
            self.log(f"Time for epoch {epoch}: {delta.seconds} s")
            self.log(f"Mean training loss for epoch {epoch}: {train_loss_mean}")
                         

    def validate(self):
        # TODO: include dynamic programming here
        # TODO: include F1 and MCC scores
        # TODO: save parameters with best F1 and MCC scores, not best loss
        # self.model.cpu()
        self.model.eval()
        val_loss_tot = 0.
        for seq, lab in self.validation_loader:
            pred = self.model(seq)
            loss = self.criterion(pred, lab)
            val_loss_tot += loss
        val_loss_mean = val_loss_tot / len(val_loader)
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": best_val_loss},
                       self.validation_file)
        # self.model.cuda()
        self.model.train()
        self.log(f"Mean validation loss for epoch {epoch}: {val_loss_mean}\n")


    @classmethod
    def from_namespace(cls, args: argparse.Namespace):
        model = nn.Sequential(*[eval(layer) for layer in args.layers])
        optimizer = eval(args.optimizer)
        criterion = eval(args.criterion)

        if torch.cuda.is_available() and not args.disable_cuda:
            device = torch.device("cuda")
            if args.copy_to_gpu:
                data_device = torch.device("cuda")
            else:
                data_device = torch.device("cpu")
        else:
            data_device = device = torch.device("cpu")

        if args.training_filetype == "bpseq-lst":
            training_set = BpseqDataset(list_file=args.training_file,
                                        device=data_device,
                                        quiet=args.quiet,
                                        batch_load=args.batch_load)
        elif args.training_filetype == "pt-lst":
            training_set = EmbeddedDataset(list_file=args.training_file,
                                           device=data_device,
                                           quiet=args.quiet,
                                           batch_load=args.batch_load)
        elif args.training_filetype == "st-lst":
            training_set = StDataset(list_file=args.training_file,
                                     device=data_device,
                                     quiet=args.quiet,
                                     batch_load=args.batch_load)
        elif args.training_filetype == "pt":
            training_set = torch.load(args.training_file)
        else:
            raise ValueError("Invalid file type")

        if hasattr(args, "validation_file"):
            if args.validation_filetype == "bpseq-lst":
                validation_set = BpseqDataset(list_file=args.validation_file,
                                              device=data_device,
                                              batch_load=args.batch_load,
                                              quiet=args.quiet)
            elif args.validation_filetype == "pt-lst":
                validation_set = EmbeddedDataset(list_file=args.validation_file,
                                                 device=data_device,
                                                 batch_load=args.batch_load,
                                                 quiet=args.quiet)
            elif args.validation_filetype == "st-lst":
                validation_set = StDataset(args.validation_file,
                                           device=data_device,
                                           batch_load=args.batch_load,
                                           quiet=args.quiet)
            elif args.validation_filetype == "pt":
                validation_set = torch.load(args.validation_file)
            else:
                raise ValueError("Invalid file type")
        else:
            validation_set = None
    
        return cls(model=model, 
                   device=device,
                   optimizer=optimizer,
                   criterion=criterion,
                   training_set=training_set,
                   validation_set=validation_set,
                   log_file=args.log_file,
                   training_checkpoint_file=args.training_checkpoint_file,
                   validation_checkpoint_file=args.validation_checkpoint_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   show_bar=args.show_bar,
                   disable_cuda=args.disable_cuda,
                   quiet=args.quiet)
