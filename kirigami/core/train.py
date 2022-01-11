import json
import sys
import time
import logging
import datetime
from functools import partial
from tqdm import tqdm
from munch import munchify, Munch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn.functional import binary_cross_entropy

import kirigami.nn
from kirigami.nn.utils import *
from kirigami.nn import WeightLoss
from kirigami.utils import *
from kirigami.utils.sampler import *


def train(notes=None, **kwargs) -> None:
    p = munchify(kwargs)
    logging.basicConfig(format="%(message)s",
                        stream=sys.stdout,
                        level=logging.INFO)
    logging.info("Run with config:\n" + str(kwargs).replace(", ", ",\n"))
    if notes:
        print("*****N.B.*****", notes)
    g = Munch() # stores "global" for training

    ####### construct all objects for training ####### 
    DEVICE = torch.device(p.device)
    # build model from `Module`'s
    module_list = [eval(layer) for layer in p.layers]
    module_list = sequentialize(module_list)
    model = nn.Sequential(*module_list)
    model = model.to(DEVICE)
    N = torch.cuda.device_count()
    # if DEVICE == torch.device("cuda") and N > 1:
    #     model = nn.DataParallel(model, output_device=[1])
    # load data and copy into `DataLoader`'s
    tr_set = torch.load(p.tr_set)
    vl_set = torch.load(p.vl_set)
    if p.batch_sample:
        g.SAMPLER = EqualSampler(tr_set)
        g.BATCH_SIZE = 1
    else:
        g.SAMPLER = None
        g.BATCH_SIZE = p.batch_size
    g.MODEL = model
    g.DEVICE = DEVICE
    g.BEST_MCC = -float("inf")
    g.BEST_LOSS = float("inf") 
    g.CRIT = eval(p.criterion)
    g.OPT = eval(p.optimizer)(g.MODEL.parameters(), lr=p.lr)
    g.SCALER = GradScaler()
    g.TR_LOADER = DataLoader(tr_set,
                             batch_size=g.BATCH_SIZE,
                             batch_sampler=g.SAMPLER,
                             collate_fn=partial(collate_fn, device=g.DEVICE))
    g.TR_LEN = len(tr_set)
    g.VL_LOADER = DataLoader(vl_set, collate_fn=partial(collate_fn, device=g.DEVICE))
    g.VL_LEN = len(vl_set)
    g.START_EPOCH = 0
    g.TR_LOSS_HISTORY = {}
    g.VL_LOSS_HISTORY = {}
    g.MCC_HISTORY = {}


    # reload model if resume
    if p.resume:
        checkpoint = torch.load(p.tr_chk) 
        g.MODEL.load_state_dict(checkpoint["model_state_dict"])
        g.OPT.load_state_dict(checkpoint["optimizer_state_dict"])
        g.START_EPOCH = checkpoint["epoch"] + 1
        g.BEST_MCC = checkpoint["best_mcc"]
        logging.info(f"Resuming at epoch {g.START_EPOCH+1} with best MCC {g.BEST_MCC}")
        

    ####### main training (and validation) loop ####### 
    for epoch in range(g.START_EPOCH, p.epochs):
        start = datetime.datetime.now()
        print(start)
        logging.info(f"Beginning epoch {epoch}")
        loss_tot = 0.

        for i, (fasta, thermo, con) in enumerate(tqdm(g.TR_LOADER, disable=not p.bar)):
            if p.thermo:
                thermo = thermo.unsqueeze(1)
                ipt = torch.cat((thermo, fasta), 1)
            else:
                ipt = fasta

            with autocast(enabled=p.mix_prec):
                if p.chkpt_seg > 0:
                    pred = checkpoint_sequential(g.MODEL, p.chkpt_seg, ipt)
                else:
                    pred = g.MODEL(ipt)
                pred = (pred + torch.transpose(pred, 2, 3)) / 2
                con = con.reshape_as(pred)
                loss = g.CRIT(pred, con)
                loss /= p.iter_acc
            g.SCALER.scale(loss).backward()

            if i % p.iter_acc == 0:
                g.SCALER.step(g.OPT)
                g.SCALER.update()
                g.OPT.zero_grad(set_to_none=True)
            loss_tot += loss.item()

        loss_avg = loss_tot / len(g.TR_LOADER) 
        g.TR_LOSS_HISTORY[epoch] = loss_avg 
        if p.tr_chk:
            torch.save({"epoch": epoch,
                        "model_state_dict": g.MODEL.state_dict(),
                        "optimizer_state_dict": g.OPT.state_dict(),
                        "best_mcc": g.BEST_MCC,
                        "cur_loss": loss_avg},
                       p.tr_chk)
        end = datetime.datetime.now()
        delta = end - start
        mess = (f"Training time for epoch {epoch}: {delta.seconds}s\n" +
                f"Mean training loss for epoch {epoch}: {loss_avg}\n" +
                f"Memory allocated: {torch.cuda.memory_allocated() / 2**20} MB\n" +
                f"Memory cached: {torch.cuda.memory_cached() / 2**20} MB")
        logging.info(mess)


        ######## validation #########

        if epoch % p.eval_freq > 0:
            print("\n\n")
            continue

        start = datetime.datetime.now()
        g.MODEL.eval()

        raw_loss_mean = bin_loss_mean = mcc_mean = 0
        f1_mean = prd_pairs_mean = grd_pairs_mean = 0
        with torch.no_grad():
            for i, (fasta, thermo, con) in enumerate(tqdm(g.VL_LOADER, disable=not p.bar)):
                if p.thermo:
                    thermo = thermo.unsqueeze(1)
                    ipt = torch.cat((thermo, fasta), 1)
                else:
                    ipt = fasta

                prd = g.MODEL(ipt)
                prd = (prd + torch.transpose(prd, 2, 3)) / 2
                con = con.reshape_as(prd)
                raw_loss = g.CRIT(prd, con)

                # used for mixed precision only (PyTorch doesn't tolerate 
                # calling BCELoss directly with mixed precision. See documention
                # re. `BCEWithLogitsLoss`)
                if isinstance(g.CRIT, torch.nn.BCEWithLogitsLoss):
                    raw_pred = torch.nn.functional.sigmoid(raw_pred) 

                seq = fasta2str(fasta)
                grd_set = grd2set(con)
                prd_set = prd2set(ipt=prd,
                                  seq=seq,
                                  thres_pairs=len(grd_set),
                                  min_dist=4,
                                  min_prob=.5)
                bin_scores = get_scores(prd_set, grd_set, len(seq))

                raw_loss_mean += raw_loss.item() / g.VL_LEN
                mcc_mean += bin_scores["mcc"] / g.VL_LEN
                f1_mean += bin_scores["f1"] / g.VL_LEN
                prd_pairs_mean += bin_scores["n_prd"] / g.VL_LEN
                grd_pairs_mean += bin_scores["n_grd"] / g.VL_LEN

        g.MCC_HISTORY[epoch] = mcc_mean
        g.VL_LOSS_HISTORY[epoch] = raw_loss_mean
        delta = datetime.datetime.now() - start
        mess = (f"Validation time for epoch {epoch}: {delta.seconds}s\n" +
                f"Raw mean validation loss for epoch {epoch}: {raw_loss_mean}\n" +
                f"Mean MCC for epoch {epoch}: {mcc_mean}\n" +
                f"Mean ground pairs: {grd_pairs_mean}\n" +
                f"Mean predicted pairs: {prd_pairs_mean}\n")
        if mcc_mean > g.BEST_MCC:
            g.BEST_MCC = mcc_mean
            g.BEST_LOSS = bin_loss_mean
            if p.vl_chk:
                torch.save({"epoch": epoch,
                            "model_state_dict": g.MODEL.state_dict(),
                            "optimizer_state_dict": g.OPT.state_dict(),
                            # "tr_loss_history": g.TR_LOSS_HISTORY,
                            # "vl_loss_history": g.VL_LOSS_HISTORY,
                            "mcc_history": g.MCC_HISTORY,
                            "grd_pairs_mean": grd_pairs_mean,
                            "prd_pairs_mean": prd_pairs_mean,
                            "mcc_mean": mcc_mean,
                            "f1_mean": f1_mean,
                            "raw_loss_mean": raw_loss_mean},
                           p.vl_chk)
            mess += f"*****NEW MAXIMUM MCC*****\n"
        mess += "\n\n"
        logging.info(mess)
        g.MODEL.train()
