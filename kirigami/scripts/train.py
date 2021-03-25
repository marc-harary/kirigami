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

import kirigami.nn
from kirigami._globals import *
from kirigami.utils.data import BpseqDataset, EmbeddedDataset, StDataset
from kirigami.utils.convert import *


__all__ = ["Train"] 


class Train:
    def __init__(self,
                 model_device: torch.device,
                 data_device: torch.device,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion: Callable,
                 epochs: int,
                 training_set: Dataset,
                 validation_set: Union[Dataset,None],
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
        self.epochs = epochs

        self.training_loader = DataLoader(training_set, shuffle=shuffle, batch_size=batch_size)
        if validation_set:
            self.validation_loader = DataLoader(validation_set, shuffle=shuffle, batch_size=batch_size)
            self.best_mean_mcc = float("inf")

        self.log_file = log_file
        self.training_checkpoint_file = training_checkpoint_file
        self.validation_checkpoint_file = validation_checkpoint_file
        self.disable_cuda = disable_cuda
        self.quiet = quiet
        self.show_bar = show_bar

        self.data_device = data_device
        self.model_device = model_device
        self.model.to(model_device)


    def log(self, message: str):
        if self.quiet:
            return
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO)
        logging.info(" " + message)


    def run(self, resume: bool = False):
        self.log("Starting training at " + str(datetime.datetime.now()))
            
        self.model.train()
        start_epoch = 0

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
            # inner_loop = tqdm(self.training_loader) if self.show_bar else self.training_loader
            
            for seq, lab in self.training_loader:
                seq_moved = seq.to(self.model_device)       
                lab_moved = lab.to(self.model_device)
                pred = self.model(seq_moved)
                loss = self.criterion(pred, lab_moved)
                train_loss_tot += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                del seq_moved, lab_moved, pred, loss
            train_loss_mean = train_loss_tot / len(self.training_loader)
            state_dict = self.model.state_dict()
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss_mean},
                        self.training_checkpoint_file)

            if self.validation_loader:
                self.validate(epoch)

            end = datetime.datetime.now()
            delta = end - start
            self.log(f"Mean training loss for epoch {epoch}: {train_loss_mean}")
            self.log(f"Time for epoch {epoch}: {delta.seconds} s\n")
                         

    def validate(self, epoch: int) -> None:
        # TODO: include dynamic programming here
        self.model.eval()
        mean_loss = 0.
        mean_mcc = 0.
        mean_f1 = 0.
        with torch.no_grad():
            for seq, lab in self.validation_loader:
                if self.model_device != self.data_device:
                    seq_moved = seq.to(self.model_device)       
                    lab_moved = lab.to(self.model_device)
                pred = self.model(seq_moved)
                loss = self.criterion(pred, lab_moved)
                pred_pair_map = tensor2pairmap(pred)
                lab_pair_map = tensor2pairmap(lab_moved)
                scores = get_scores(pred_pair_map, lab_pair_map)
                mean_mcc += scores[4]
                mean_f1 += scores[5] 
                mean_loss += loss
                del seq_moved, lab_moved, pred, loss
        mean_loss /= len(self.validation_loader)
        mean_f1 /= len(self.validation_loader)
        mean_mcc /= len(self.validation_loader) 
        if mean_mcc < self.best_mean_mcc:
            self.best_mean_mcc = mean_mcc
            self.log(f"New optimum at epoch {epoch}")
            self.best_validation_loss = mean_loss
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "mean_mcc": mean_mcc,
                        "mean_f1": mean_f1,
                        "loss": self.best_validation_loss},
                       self.validation_checkpoint_file)
        self.log(f"Mean validation loss for epoch {epoch}: {mean_loss}")
        self.log(f"Mean MCC for epoch {epoch}: {mean_mcc}")
        self.log(f"Mean F1 for epoch {epoch}: {mean_f1}")
        self.model.train()


    @classmethod
    def from_namespace(cls, args: argparse.Namespace):
        if args.model_device == "gpu":
            if torch.cuda.is_available():
                model_device = torch.device("cuda")
            else:
                raise ValueError("CUDA is not available")
        else:
            model_device = torch.device("cpu")

        if args.data_device == "gpu":
            if torch.cuda.is_available():
                data_device = torch.device("cuda")
            else:
                raise ValueError("CUDA is not available")
        else:
            data_device = torch.device("cpu")
        
        if args.training_filetype == "bpseq-lst":
            training_set = BpseqDataset(list_file=args.training_file,
                                        device=data_device,
                                        batch_load=args.batch_load)
        elif args.training_filetype == "pt-lst":
            training_set = EmbeddedDataset(list_file=args.training_file,
                                           device=data_device,
                                           batch_load=args.batch_load)
        elif args.training_filetype == "st-lst":
            training_set = StDataset(list_file=args.training_file,
                                     device=data_device,
                                     batch_load=args.batch_load)
        elif args.training_filetype == "pt":
            training_set = torch.load(args.training_file)
            if training_set.device != data_device:
                training_set.to(data_device)
        else:
            raise ValueError("Invalid file type")

        if hasattr(args, "validation_file"):
            if args.validation_filetype == "bpseq-lst":
                validation_set = BpseqDataset(list_file=args.validation_file,
                                              device=data_device,
                                              batch_load=args.batch_load)
            elif args.validation_filetype == "pt-lst":
                validation_set = EmbeddedDataset(list_file=args.validation_file,
                                                 device=data_device,
                                                 batch_load=args.batch_load)
            elif args.validation_filetype == "st-lst":
                validation_set = StDataset(args.validation_file,
                                           device=data_device,
                                           batch_load=args.batch_load)
            elif args.validation_filetype == "pt":
                validation_set = torch.load(args.validation_file)
                if validation_set.device != data_device:
                    validation_set.to(data_device)
            else:
                raise ValueError("Invalid file type")
        else:
            validation_set = None

        model = nn.Sequential(*[eval(layer) for layer in args.layers])
        model.to(model_device)
        optimizer = eval(args.optimizer)
        criterion = eval(args.criterion)

        return cls(model=model, 
                   model_device=model_device,
                   data_device=data_device,
                   optimizer=optimizer,
                   criterion=criterion,
                   epochs=args.epochs,
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
