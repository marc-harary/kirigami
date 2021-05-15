import os
import itertools
import time
import logging
import datetime
from typing import Union, Callable, List, Dict, Optional, Tuple
from pathlib import Path
import gc

import argparse
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import torch.multiprocessing as mp

import kirigami.nn
from kirigami._globals import *
from kirigami.utils.data import BpseqDataset, EmbeddedDataset, StDataset
from kirigami.utils.convert import *
from kirigami.utils.process import *
from kirigami.utils.memory_utils import *
from kirigami import binarize
from kirigami.utils import *


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
    checkpoint_segments: int
    training_loader: DataLoader
    validation_loader: Union[DataLoader,None]
    validation_device: torch.device
    log_file: Union[Path,None]
    training_checkpoint_file: Union[Path,None]
    validation_checkpoint_file: Union[Path,None]
    thres_prob: float
    thres_by_ground_pairs: bool
    canonicalize: bool
    symmetrize: bool
    show_batch_bar: bool
    show_epoch_bar: bool
    pre_concatenate: bool
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
                 checkpoint_segments: int,
                 training_loader: DataLoader,
                 validation_loader: Union[DataLoader,None],
                 validation_device: torch.device,
                 log_file: Union[Path,None],
                 training_checkpoint_file: Union[Path,None],
                 validation_checkpoint_file: Union[Path,None],
                 checkpoint_gradients: bool = False,
                 thres_prob: float = 0.5,
                 thres_by_ground_pairs: bool = True,
                 canonicalize: bool = True,
                 symmetrize: bool = True,
                 show_batch_bar: bool = True,
                 show_epoch_bar: bool = True,
                 pre_concatenate: bool = True,
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
        self.checkpoint_segments = checkpoint_segments

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.validation_device = validation_device
        self.best_mcc = None if not validation_loader else -1.
        self.best_loss = None if not validation_loader else float("inf")

        self.thres_prob = thres_prob
        self.thres_by_ground_pairs = thres_by_ground_pairs
        self.canonicalize = canonicalize
        self.symmetrize = symmetrize
        self.pre_concatenate = pre_concatenate

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
        self.show_batch_bar = show_batch_bar
        self.show_epoch_bar = show_epoch_bar


    def log(self, message: str) -> None:
        if not self.quiet:
            logging.info(" " + message)


    def collate(self, seq: torch.Tensor, lab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lab_out = lab.to_dense() if lab.is_sparse else lab
        seq_out = seq.to_dense() if seq.is_sparse else seq
        if seq_out.device != self.model_device:
            seq_out = seq_out.to(self.model_device)
            lab_out = lab_out.to(self.model_device)
        if self.pre_concatenate:
            seq_out = kirigami.concatenate_batch(seq_out)
        if seq_out.dtype != torch.float64:
            seq_out = seq_out.float()
            lab_out = lab_out.float()
        return seq_out, lab_out


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

        epoch_iterator = range(start_epoch, self.epochs)
        epoch_loop = tqdm(epoch_iterator) if self.show_epoch_bar else epoch_iterator

        scaler = GradScaler()

        for epoch in epoch_loop:
            start = datetime.datetime.now()
            self.log(f"Starting epoch {epoch} at " + str(start))
            length = len(self.training_loader)
            train_loss_tot = 0.
            batch_iterator = enumerate(self.training_loader)
            batch_loop = tqdm(batch_iterator) if self.show_batch_bar else batch_iterator
            for i, batch in batch_loop:
                # seq_copy = concatenate_batch(seq).to(self.model_device).float()
                seq_coll, lab_coll = self.collate(*batch)
                with autocast(enabled=self.mixed_precision):
                    if self.checkpoint_segments > 0:
                        pred = torch.utils.checkpoint.checkpoint_sequential(self.model, self.checkpoint_segments, seq_coll)
                    else:
                        pred = self.model(seq_coll)
                    loss = self.criterion(pred, lab_coll.reshape_as(pred))
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
            self.log("Memory allocated: " + str(torch.cuda.memory_allocated() / 2**20) + " MB")
            self.log("Memory cached: " + str(torch.cuda.memory_cached() / 2**20) + " MB\n")


    def validate(self, epoch: int) -> None:
        start = datetime.datetime.now()
        self.model.eval()

        n_batch = len(self.validation_loader)
        raw_mean_loss = mean_loss = mean_mcc = mean_f1 = mean_pairs = 0.
        mean_ground_pairs = 0

        batch_iterator = enumerate(self.validation_loader)
        batch_loop = tqdm(batch_iterator) if self.show_batch_bar else batch_iterator

        with torch.no_grad():
            for i, (seq, lab) in batch_loop:
                seq_length = seq.sum().item()
                seq_coll, lab_coll = self.collate(seq, lab)
                lab_pair_map = dense2pairmap(lab_coll, seq_length=seq_length)
                ground_pairs = int(lab.sum().item() / 2)
                mean_ground_pairs += ground_pairs / n_batch

                raw_pred = self.model(seq_coll)
                raw_loss = self.criterion(raw_pred, lab_coll.reshape_as(raw_pred))
                raw_mean_loss += raw_loss.item() / n_batch
                if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    raw_pred = torch.nn.functional.sigmoid(raw_pred) 

                bin_pred = binarize(seq=dense2sequence(seq),
                                    lab=lab_coll,
                                    thres_prob=self.thres_prob,
                                    thres_pairs=ground_pairs if self.thres_by_ground_pairs else sys.maxsize,
                                    symmetrize=self.symmetrize,
                                    canonicalize=self.canonicalize)
                if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    bin_loss = torch.nn.functional.binary_cross_entropy(bin_pred, lab_coll.reshape_as(bin_pred))
                else:
                    bin_loss = self.criterion(bin_pred, lab_coll)

                pair_map = dense2pairmap(bin_pred, seq_length=seq_length)
                scores = get_scores(pair_map, lab_pair_map)
                mean_mcc += scores.mcc / n_batch
                mean_f1 += scores.f1 / n_batch
                mean_pairs += scores.pred_pairs / n_batch
                mean_loss += bin_loss.item() / n_batch

        if raw_mean_loss < self.best_loss:
            self.log(f"New optimum at epoch {epoch}")
            self.best_mcc = mean_mcc
            self.best_loss = raw_mean_loss
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "mean_ground_pairs": mean_ground_pairs,
                        "mean_pred_pairs": mean_pairs,
                        "mean_mcc": mean_mcc,
                        "mean_f1": mean_f1,
                        "mean_loss": mean_loss,
                        "raw_mean_loss": raw_mean_loss,
                        "best_loss": self.best_loss},
                       self.validation_checkpoint_file)
        end = datetime.datetime.now()
        delta = end - start
        self.log(f"Validation time for epoch {epoch}: {delta.seconds} s")
        self.log(f"Raw mean validation loss for epoch {epoch}: {raw_mean_loss}")
        self.log(f"Binarized mean validation loss for epoch {epoch}: {mean_loss}")
        self.log(f"Mean MCC for epoch {epoch}: {mean_mcc}")
        self.log(f"Mean F1 for epoch {epoch}: {mean_f1}")
        self.log(f"Mean predicted pairs for epoch {epoch}: {mean_pairs}")
        self.log(f"Actual mean pairs: {mean_ground_pairs}")
        self.model.to(self.model_device)
        self.model.train()


    @classmethod
    def from_namespace(cls, args: argparse.Namespace):
        if args.model_device == "cuda":
            if torch.cuda.is_available():
                model_device = torch.device("cuda")
            else:
                raise ValueError("CUDA is not available")
        else:
            model_device = torch.device("cpu")
        if args.training_data_device == "cuda":
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
            if args.validation_data_device == "cuda":
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

        module_list = []
        for layer in args.layers:
            module_list.extend(flatten_module(eval(layer)))
        model = nn.Sequential(*module_list)
        
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
                   checkpoint_segments=args.checkpoint_segments,
                   training_loader=training_loader,
                   validation_loader=validation_loader,
                   validation_device=validation_data_device,
                   log_file=args.log_file,
                   training_checkpoint_file=args.training_checkpoint_file,
                   validation_checkpoint_file=args.validation_checkpoint_file,
                   thres_prob=args.thres_prob,
                   thres_by_ground_pairs=args.thres_by_ground_pairs,
                   pre_concatenate=not args.disable_pre_concatenation,
                   show_batch_bar=not args.disable_batch_bar,
                   show_epoch_bar=not args.disable_epoch_bar,
                   canonicalize=not args.disable_canonicalize,
                   symmetrize=not args.disable_symmetrize,
                   quiet=args.quiet)
