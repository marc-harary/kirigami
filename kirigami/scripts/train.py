import os
import time
import logging
import datetime
from typing import Union, Callable, List, Dict, Optional
from pathlib import Path
import gc

import argparse
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
# import torch.multiprocessing as mp

import kirigami.nn
from kirigami._globals import *
from kirigami.utils.data import BpseqDataset, EmbeddedDataset, StDataset
from kirigami.utils.convert import *
from kirigami.utils.process import *
from kirigami.utils.memory_utils import *
# from kirigami import binarize


__all__ = ["Train"] 


class Train:
    """Namespace for training scripts.""" 

    model: nn.Module
    model_device: torch.device
    optimizer: torch.optim
    mixed_precision: bool
    criterion: Callable
    epochs: int
    iters_to_accumulate: int
    segments: int
    training_loader: DataLoader
    validation_loader: Union[DataLoader,None]
    validation_device: torch.device
    log_file: Union[Path,None]
    training_checkpoint_file: Union[Path,None]
    validation_checkpoint_file: Union[Path,None]
    checkpoint_gradients: bool
    binarize: bool
    thres: float
    canonicalize: bool
    symmetrize: bool
    show_bar: bool
    quiet: bool
    best_mcc: Optional[float]
    best_loss: Optional[float]
    logs: List[Dict]
    hooks: List[torch.utils.hooks.RemovableHandle]
    
    def __init__(self,
                 model: nn.Module,
                 model_device: torch.device,
                 optimizer: torch.optim,
                 mixed_precision: bool,    
                 criterion: Callable,
                 epochs: int,
                 iters_to_accumulate: int,
                 segments: int,
                 training_loader: DataLoader,
                 validation_loader: Union[DataLoader,None],
                 validation_device: torch.device,
                 log_file: Union[Path,None],
                 training_checkpoint_file: Union[Path,None],
                 validation_checkpoint_file: Union[Path,None],
                 checkpoint_gradients: bool = False,
                 binarize: bool = True,
                 thres: float = 0.5,
                 canonicalize: bool = True,
                 symmetrize: bool = True,
                 show_bar: bool = True,
                 quiet: bool = False) -> None:
        self.model = model
        self.model_device = model_device
        self.scaler = mixed_precision
        self.optimizer = optimizer
        self.criterion = criterion
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.epochs = epochs
        self.iters_to_accumulate = iters_to_accumulate
        self.checkpoint_gradients = checkpoint_gradients
        self.segments = segments

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.validation_device = validation_device
        self.best_mcc = None if not validation_loader else -1.
        self.best_loss = None if not validation_loader else float("inf")

        self.binarize = binarize
        self.thres = thres
        self.canonicalize = canonicalize
        self.symmetrize = symmetrize

        self.training_checkpoint_file = training_checkpoint_file
        self.validation_checkpoint_file = validation_checkpoint_file
        self.log_file = log_file
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO)
    
        self.logs = []
        self.hooks = []
        # for idx, module in enumerate(self.model.modules()):
        #     self.hooks.append(module.register_forward_pre_hook(make_hook(self.logs, idx, "pre")))
        #     self.hooks.append(module.register_forward_hook(make_hook(self.logs, idx, "fwd")))
        #     self.hooks.append(module.register_backward_hook(make_hook(self.logs, idx, "bwd")))

        self.quiet = quiet
        self.show_bar = show_bar


    def log(self, message: str) -> None:
        if not self.quiet:
            logging.info(" " + message)


    def run(self, resume: bool = False) -> None:
        self.log("Starting training at " + str(datetime.datetime.now()))
        self.log("Initial memory cached: " + str(torch.cuda.memory_cached() / 2**20))
        self.log("Initial memory allocated: " + str(torch.cuda.memory_allocated() / 2**20) + "\n")

        self.model.train()
        start_epoch = 0

        if resume:
            checkpoint = torch.load(self.validation_checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            self.log(f"Resuming at epoch {start_epoch} with loss {loss}")

        range_iterator = range(start_epoch, self.epochs)
        outer_loop = tqdm(range_iterator) if self.show_bar else range_iterator

        scaler = GradScaler()

        for epoch in outer_loop:
            start = datetime.datetime.now()
            self.log(f"Starting epoch {epoch} at " + str(start))
            length = len(self.training_loader)
            train_loss_tot = 0.
            for i, (seq, lab) in tqdm(enumerate(self.training_loader)):
                seq_copy = concatenate_batch(seq).to(self.model_device).float()
                lab_copy = lab.to(self.model_device).float()
                with autocast(enabled=self.mixed_precision):
                    if self.checkpoint_gradients:
                        # pred = torch.utils.checkpoint.checkpoint_sequential(self.model, self.segments, seq.to(self.model_device).float())
                        pred = torch.utils.checkpoint.checkpoint_sequential(self.model, self.segments, seq_copy)
                    else:
                        # pred = self.model(seq.to(self.model_device).float()) 
                        pred = self.model(seq_copy)
                    # loss = self.criterion(pred, lab.to(self.model_device).float())
                    loss = self.criterion(pred, lab_copy)
                    loss /= self.iters_to_accumulate
                scaler.scale(loss).backward()
                if i % self.iters_to_accumulate == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                train_loss_tot += loss.item()
            train_loss_mean = train_loss_tot / len(self.training_loader)
            state_dict = self.model.state_dict()
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_loss_mean},
                        self.training_checkpoint_file)
            end = datetime.datetime.now()
            delta = end - start
            self.log(f"Training time for epoch {epoch}: {delta.seconds} s")
            self.log(f"Mean training loss for epoch {epoch}: {train_loss_mean}")
            if self.validation_loader:
                self.validate(epoch)
            self.log("Memory allocated: " + str(torch.cuda.memory_allocated() / 2**20))
            self.log("Memory cached: " + str(torch.cuda.memory_cached() / 2**20) + "\n")


    def validate(self, epoch: int) -> None:
        start = datetime.datetime.now()
        self.model.eval()

        n_batch = len(self.validation_loader)
        raw_mean_loss = raw_mean_mcc = raw_mean_f1 = raw_mean_pairs = 0.
        bin_mean_loss= bin_mean_mcc = bin_mean_f1 = bin_mean_pairs = 0. if self.binarize else None
        mean_ground_pairs = 0

        with torch.no_grad():
            for i, (seq, lab) in enumerate(self.validation_loader):
                seq_length = seq.sum().item()
                seq_copy = concatenate_batch(seq).to(self.model_device).float()
                lab_copy = lab.to(self.model_device).float()
                lab_pair_map = dense2pairmap(lab, seq_length=seq_length)

                raw_pred = self.model(seq_copy)
                raw_loss = self.criterion(raw_pred, lab_copy)
                if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    raw_pred = torch.nn.functional.sigmoid(raw_pred) 
                raw_pair_map = dense2pairmap(raw_pred, seq_length=seq_length)
                raw_scores = get_scores(raw_pair_map, lab_pair_map)
                raw_mean_mcc += raw_scores.mcc / n_batch
                raw_mean_f1 += raw_scores.f1 / n_batch
                raw_mean_pairs += raw_scores.pred_pairs / n_batch
                raw_mean_loss += raw_loss.item() / n_batch

                mean_ground_pairs += raw_scores.ground_pairs / n_batch

                if self.binarize:
                    bin_pred = binarize(seq,
                                        raw_pred,
                                        max_pad=512,
                                        thres=self.thres,
                                        symmetrize=self.symmetrize,
                                        canonicalize=self.canonicalize)
                    bin_loss = self.criterion(bin_pred, lab_copy)
                    bin_pair_map = dense2pairmap(bin_pred, seq_length=seq_length)
                    bin_scores = get_scores(bin_pair_map, lab_pair_map)
                    bin_mean_mcc += bin_scores.mcc / n_batch
                    bin_mean_f1 += bin_scores.f1 / n_batch
                    bin_mean_pairs += bin_scores.pred_pairs / n_batch
                    bin_mean_loss += bin_loss.item() / n_batch

        better_mcc = max(raw_mean_mcc, bin_mean_mcc) 
        better_loss = min(raw_mean_loss, bin_mean_loss)

        # if better_mean_loss < self.best_loss:
        if better_mcc > self.best_mcc:
            self.log(f"New optimum at epoch {epoch}")
            self.best_mcc = better_mcc
            self.best_loss = min(self.best_loss, better_loss)
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "raw_mean_loss": raw_mean_loss,
                        "raw_mean_mcc": raw_mean_mcc,
                        "raw_mean_f1": raw_mean_f1,
                        "raw_mean_pred_pairs": raw_mean_pairs,
                        "bin_mean_loss": bin_mean_loss,
                        "bin_mean_mcc": bin_mean_mcc,
                        "bin_mean_f1": bin_mean_f1,
                        "bin_mean_pred_pairs": bin_mean_pairs,
                        "best_loss": self.best_loss},
                       self.validation_checkpoint_file)
        end = datetime.datetime.now()
        delta = end - start
        self.log(f"Validation time for epoch {epoch}: {delta.seconds} s")
        self.log(f"Mean raw validation loss for epoch {epoch}: {raw_mean_loss}")
        self.log(f"Mean raw MCC for epoch {epoch}: {raw_mean_mcc}")
        self.log(f"Mean raw F1 for epoch {epoch}: {raw_mean_f1}")
        self.log(f"Mean binarized validation loss for epoch {epoch}: {bin_mean_loss}")
        self.log(f"Mean binarized MCC for epoch {epoch}: {bin_mean_mcc}")
        self.log(f"Mean binarized F1 for epoch {epoch}: {bin_mean_f1}")
        self.model.to(self.model_device)
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
        if args.training_data_device == "gpu":
            if torch.cuda.is_available():
                training_data_device = torch.device("cuda")
            else:
                raise ValueError("CUDA is not available")
        else:
            training_data_device = torch.device("cpu")
        
        if args.training_filetype == "bpseq-lst":
            training_set = BpseqDataset(list_file=args.training_file,
                                        max_len=args.max_length,
                                        device=training_data_device,
                                        batch_load=args.batch_load)
        elif args.training_filetype == "pt-lst":
            training_set = EmbeddedDataset(list_file=args.training_file,
                                           max_len=args.max_length,
                                           device=training_data_device,
                                           batch_load=args.batch_load)
        elif args.training_filetype == "st-lst":
            training_set = StDataset(list_file=args.training_file,
                                     max_len=args.max_length,
                                     device=training_data_device,
                                     batch_load=args.batch_load)
        elif args.training_filetype == "pt":
            training_set = torch.load(args.training_file)
        else:
            raise ValueError("Invalid file type")
        training_loader = DataLoader(training_set,
                                     shuffle=args.shuffle,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=(training_data_device == torch.device("cpu")))

        validation_loader = None
        if hasattr(args, "validation_file"):
            if args.validation_data_device == "gpu":
                if torch.cuda.is_available():
                    validation_data_device = torch.device("cuda")
                else:
                    raise ValueError("CUDA is not available")
            else:
                validation_data_device = torch.device("cpu")
            if args.validation_filetype == "bpseq-lst":
                validation_set = BpseqDataset(list_file=args.validation_file,
                                              max_len=args.max_length,
                                              device=validation_data_device,
                                              batch_load=args.batch_load)
            elif args.validation_filetype == "pt-lst":
                validation_set = EmbeddedDataset(list_file=args.validation_file,
                                                 max_len=args.max_length,
                                                 device=validation_data_device,
                                                 batch_load=args.batch_load)
            elif args.validation_filetype == "st-lst":
                validation_set = StDataset(args.validation_file,
                                           max_len=args.max_length,
                                           device=validation_data_device,
                                           batch_load=args.batch_load)
            elif args.validation_filetype == "pt":
                validation_set = torch.load(args.validation_file)
            else:
                raise ValueError("Invalid file type")
            validation_loader = DataLoader(validation_set,
                                           shuffle=False,
                                           batch_size=1)

        model = nn.Sequential(*[eval(layer) for layer in args.layers])
        
        model = model.to(model_device)
        N = torch.cuda.device_count()
        if model_device == torch.device("cuda") and N > 1:
            model = nn.DataParallel(model, output_device=[1])

        optimizer = eval(args.optimizer)
        criterion = eval(args.criterion)

        return cls(model=model, 
                   model_device=model_device,
                   mixed_precision=args.mixed_precision,
                   optimizer=optimizer,
                   criterion=criterion,
                   epochs=args.epochs,
                   iters_to_accumulate=args.iters_to_accumulate,
                   segments=args.segments,
                   training_loader=training_loader,
                   validation_loader=validation_loader,
                   validation_device=validation_data_device,
                   log_file=args.log_file,
                   training_checkpoint_file=args.training_checkpoint_file,
                   validation_checkpoint_file=args.validation_checkpoint_file,
                   show_bar=args.show_bar,
                   binarize=args.binarize,
                   thres=args.thres,
                   canonicalize=args.canonicalize,
                   symmetrize=args.symmetrize,
                   checkpoint_gradients=args.checkpoint_gradients,
                   quiet=args.quiet)
