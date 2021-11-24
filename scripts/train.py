import os
import sys
import json
import itertools
from functools import reduce
import time
import logging
import datetime
from pathlib import Path
from tqdm import tqdm
from munch import munchify, Munch
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn.functional import binary_cross_entropy

import kirigami.nn
from kirigami.nn.utils import *
from kirigami.utils import binarize


# class ZukerModule(nn.Module):
#     def __init__(self, in_shape=512, out_shape=512):
#         super().__init__()
#         self._in_shape = in_shape
#         self._out_shape = out_shape
#         self.res = ResNetBlock(p=0.5,
#                                dilations=None,
#                                kernel_sizes=None,
#                                act = None,
#                                n_channels = None,
#                                resnt = True)


def concat(fasta):
    out = fasta.unsqueeze(-1)
    out = torch.cat(out.shape[-2] * [out], dim=-1)
    out_t = out.transpose(-1, -2)
    out = torch.cat([out, out_t], dim=-3)
    return out


def to_str(ipt):
    ipt_ = ipt.squeeze()
    total_length = ipt_.shape[1]
    fasta_length = int(ipt_.sum().item())
    beg = (total_length - fasta_length) // 2
    end = beg + fasta_length
    _, js = torch.max(ipt_[:,beg:end], 0)
    return "".join("AUCG"[j] for j in js)

@dataclass
class Scores:
    tp: float
    tn: float
    fp: float
    fn: float
    f1: float
    mcc: float
    ground_pairs: int
    pred_pairs: int


# def get_scores(pred, grd, length):
#     total = length * (length-1) / 2
# 
#     total_length = pred_.shape[0]
#     beg = (total_length - length) // 2
#     end = beg + length
# 
#     vals, idxs = torch.max(pred[:,beg:end], 0)
#     prd_pairs = set()
#     for i, (val, idx) in enumerate(zip(vals, idxs)):
#         if val == 1:
#             prd_pairs.add((i, idx))
# 
#     vals, idxs = torch.max(grd[:,beg:end], 0)
#     grd_pairs = set()
#     for i, (val, idx) in enumerate(zip(vals, idxs)):
#         if val == 1:
#             grd_pairs.add((i, idx))
# 
#     n_prd, n_grd = len(prd_pairs), len(grd_pairs)
#     tp = float(len(prd_pairs.intersection(grd_pairs)))
#     fp = len(prd_pairs) - tp
#     fn = len(grd_pairs) - tp
#     tn = total - tp - fp - fn
#     mcc = f1 = 0. 
#     if n_prd > 0 and n_grd > 0:
#         sn = tp / (tp+fn)
#         pr = tp / (tp+fp)
#         if tp > 0:
#             f1 = 2*sn*pr / (pr+sn)
#         if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
#             mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
#     return Scores(tp, tn, fp, fn, f1, mcc, n_grd, n_prd)


def get_scores(pred, grd):
    length = pred.shape[0]

    total = length * (length-1) / 2

    vals, idxs = torch.max(pred, 0)
    prd_pairs = set()
    for i, (val, idx) in enumerate(zip(vals, idxs)):
        if val == 1:
            prd_pairs.add((i, idx.item()))

    vals, idxs = torch.max(grd, 0)
    grd_pairs = set()
    for i, (val, idx) in enumerate(zip(vals, idxs)):
        if val == 1:
            grd_pairs.add((i, idx.item()))

    n_prd, n_grd = len(prd_pairs), len(grd_pairs)
    tp = float(len(prd_pairs.intersection(grd_pairs)))
    fp = len(prd_pairs) - tp
    fn = len(grd_pairs) - tp
    tn = total - tp - fp - fn
    mcc = f1 = 0. 
    if n_prd > 0 and n_grd > 0:
        sn = tp / (tp+fn)
        pr = tp / (tp+fp)
        if tp > 0:
            f1 = 2*sn*pr / (pr+sn)
        if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
            mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return Scores(tp, tn, fp, fn, f1, mcc, n_grd, n_prd)





def main():
    with open(sys.argv[1], "r") as f:
        txt = f.read()
        p = munchify(json.loads(txt))

    logging.basicConfig(format="%(asctime)s\n%(message)s\n",
                        stream=sys.stdout,
                        level=logging.INFO)
    logging.info("Run with config:\n"+txt) 

    # stores "global" variables (objects) for training
    g = Munch()

    ####### construct all objects for training ####### 

    DEVICE = torch.device(p.device)

    # build model from `Module`'s
    module_list = [eval(layer) for layer in p.layers]
    module_list = sequentialize(module_list)
    model = nn.Sequential(*module_list)
    model = model.to(DEVICE)
    N = torch.cuda.device_count()
    if model == torch.device("cuda") and N > 1:
        model = nn.DataParallel(model, output_device=[1])

    # load data and copy into `DataLoader`'s
    tr_set = torch.load(p.tr_set)
    tr_set.tensors = [tensor.to(DEVICE) for tensor in tr_set.tensors]
    vl_set = torch.load(p.vl_set)
    vl_set.tensors = [tensor.to(DEVICE) for tensor in vl_set.tensors]

    g.MODEL = model
    g.DEVICE = DEVICE
    g.BEST_MCC = -float("inf")
    g.BEST_LOSS = float("inf") 
    g.CRIT = eval(p.criterion)()
    g.OPT = eval(p.optimizer)(g.MODEL.parameters())
    g.SCALER = GradScaler()
    g.TR_LOADER = DataLoader(tr_set, shuffle=True, batch_size=p.batch_size)
    g.TR_LEN = len(tr_set)
    g.VL_LOADER = DataLoader(vl_set, shuffle=False, batch_size=1)
    g.VL_LEN = len(vl_set)

    ####### main training (and validation) loop ####### 

    for epoch in range(p.epochs):
        start = datetime.datetime.now()
        logging.info(f"Beginning epoch {epoch}")
        loss_tot = 0.

        for i, batch in enumerate(tqdm(g.TR_LOADER)):
            if p.thermo:
                fasta, thermo, con = batch
                # dense->sparse, uint8->float, sparse to concat 
                fasta = concat(fasta.to_dense().float())
                con = con.to_dense().float()
                thermo = thermo.to_dense()
                thermo = thermo.unsqueeze(1)
                fasta = torch.cat((thermo, fasta), 1)
            else:
                fasta, con = batch
                fasta = concat(fasta.to_dense().float())
                con = con.to_dense().float()

            with autocast(enabled=p.mix_prec):
                if p.chkpt_seg > 0:
                    pred = checkpoint_sequential(g.MODEL, p.chkpt_seg, fasta)
                else:
                    pred = g.MODEL(ipt)
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
        torch.save({"epoch": epoch,
                    "model_state_dict": g.MODEL.state_dict(),
                    "optimizer_state_dict": g.OPT.state_dict(),
                    "loss": loss_avg},
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
            continue

        start = datetime.datetime.now()
        g.MODEL.eval()

        raw_loss_mean = bin_loss_mean = mcc_mean = 0
        f1_mean = prd_pairs_mean = grd_pairs_mean = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(g.VL_LOADER)):
                if p.thermo:
                    fasta2d, thermo, con_grd = batch
                    fasta2d = fasta2d.to_dense()
                    con_grd = con_grd.to_dense().float().reshape(512, 512)
                    thermo = thermo.to_dense()
                    thermo = thermo.unsqueeze(1)
                    fasta = concat(fasta2d.float())
                    fasta = torch.cat((thermo, fasta), 1)
                else:
                    fasta, con = batch
                    fasta2d = fasta.to_dense()
                    fasta = concat(fasta2d.float())
                    con_grd = con.to_dense().float().reshape(512, 512)
                    
                raw_pred = g.MODEL(fasta).squeeze()
                raw_loss = g.CRIT(raw_pred, con_grd)

                if isinstance(g.CRIT, torch.nn.BCEWithLogitsLoss):
                    raw_pred = torch.nn.functional.sigmoid(raw_pred) 

                thres_pairs = int(con_grd.sum().item() / 2)
                seq = to_str(fasta2d)
                beg = (fasta2d.shape[-1] - len(seq)) // 2
                end = beg + len(seq)

                bin_pred = binarize(lab=raw_pred,
                                    seq=seq,
                                    min_dist=4,
                                    thres_pairs=thres_pairs,
                                    thres_prob=0,
                                    symmetrize=True,
                                    canonicalize=True)

                if isinstance(g.CRIT, torch.nn.BCEWithLogitsLoss):
                    bin_loss = binary_cross_entropy(bin_pred, con_grd)
                else:
                    bin_loss = g.CRIT(bin_pred, con_grd)

                bin_scores = get_scores(bin_pred[beg:end, beg:end], con_grd[beg:end, beg:end])
                # raw_scores = get_scores(raw_pred, con_grd, length=len(seq))

                raw_loss_mean += raw_loss.item() / g.VL_LEN
                bin_loss_mean += bin_loss.item() / g.VL_LEN
                mcc_mean += bin_scores.mcc / g.VL_LEN
                f1_mean += bin_scores.f1 / g.VL_LEN
                prd_pairs_mean += bin_scores.pred_pairs / g.VL_LEN
                grd_pairs_mean += thres_pairs / g.VL_LEN


        delta = datetime.datetime.now() - start
        mess = (f"Validation time for epoch {epoch}: {delta.seconds}s\n" +
                f"Raw mean validation loss for epoch {epoch}: {raw_loss_mean}\n" +
                f"Mean MCC for epoch {epoch}: {mcc_mean}\n" +
                f"Mean ground pairs: {grd_pairs_mean}\n" +
                f"Mean predicted pairs: {prd_pairs_mean}\n" +
                f"Binarized mean validation loss for epoch {epoch}: {bin_loss_mean}\n")
        if mcc_mean > g.BEST_MCC:
            logging.info(f"New optimum at epoch {epoch}")
            g.BEST_MCC = mcc_mean
            g.BEST_LOSS = bin_loss_mean
            torch.save({"epoch": epoch,
                        "model_state_dict": g.MODEL.state_dict(),
                        "optimizer_state_dict": g.OPT.state_dict(),
                        "grd_pairs_mean": grd_pairs_mean,
                        "prd_pairs_mean": prd_pairs_mean,
                        "mcc_mean": mcc_mean,
                        "f1_mean": f1_mean,
                        "raw_loss_mean": raw_loss_mean,
                        "bin_loss_mean": bin_loss_mean},
                       p.vl_chk)
            mess += f"*****NEW MAXIMUM MCC*****\n"
        mess += "\n\n"
        logging.info(mess)
        g.MODEL.train()


if __name__ == "__main__":
    main()
