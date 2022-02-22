import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from kirigami.nn import WeightLoss, ForkL1 #ForkLoss, ForkLogCosh
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
    print("\n\n")
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
    g.SCALER = GradScaler() if g.DEVICE == torch.device("cuda") else None
    g.TR_LOADER = DataLoader(tr_set,
                             batch_size=g.BATCH_SIZE,
                             batch_sampler=g.SAMPLER,
                             collate_fn=partial(collate_fn,
                                                use_dist=p.use_dist,
                                                n_dists=p.n_dists,
                                                use_thermo=p.use_thermo,
                                                device=g.DEVICE))
    g.TR_LEN = len(tr_set)
    g.VL_LOADER = DataLoader(vl_set, collate_fn=partial(collate_fn,
                                                        use_dist=p.use_dist,
                                                        n_dists=p.n_dists,
                                                        use_thermo=p.use_thermo,
                                                        device=g.DEVICE))
    g.VL_LEN = len(vl_set)
    g.START_EPOCH = 0


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
        dist_loss_tot = 0.
        con_loss_tot = 0.

        for i, (ipt, grd) in enumerate(tqdm(g.TR_LOADER, disable=not p.bar)):
            with autocast(enabled=p.mix_prec):
                if p.chkpt_seg > 0:
                    prd = checkpoint_sequential(g.MODEL, p.chkpt_seg, ipt)
                else:
                    prd = g.MODEL(ipt)
                loss, dist_loss, con_loss = g.CRIT(prd, grd)
                loss /= p.iter_acc
            if g.SCALER:
                g.SCALER.scale(loss).backward()
            else:
                loss.backward()

            if i % p.iter_acc == 0:
                if g.SCALER:
                    g.SCALER.step(g.OPT)
                    g.SCALER.update()
                else:
                    g.OPT.step()
                g.OPT.zero_grad(set_to_none=True)
            loss_tot += loss.item() * p.iter_acc
            dist_loss_tot += dist_loss
            con_loss_tot += con_loss

        loss_avg = loss_tot / len(g.TR_LOADER) 
        dist_loss_avg = dist_loss_tot / len(g.TR_LOADER) 
        con_loss_avg = con_loss_tot / len(g.TR_LOADER) 
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
                f"Mean total training loss for epoch {epoch}: {loss_avg}\n" +
                f"Mean distance loss for epoch {epoch}: {dist_loss_avg}\n" +
                f"Mean contact loss for epoch {epoch}: {con_loss_avg}\n" +
                f"Memory allocated: {torch.cuda.memory_allocated() / 2**20} MB\n" +
                f"Memory reserved: {torch.cuda.memory_reserved() / 2**20} MB")
        logging.info(mess)


        ######## validation #########

        if epoch % p.eval_freq > 0:
            print("\n\n")
            continue

        start = datetime.datetime.now()
        g.MODEL.eval()

        raw_loss_tot = bin_loss_tot = mcc_tot = 0
        f1_tot = prd_pairs_tot = grd_pairs_tot = 0
        ls, pccs, mae_ls, mae_dists = [], [], [], []
        con_loss_tot = 0
        dist_loss_tot = 0
        count = 0

        prd_grds = []
        with torch.no_grad():
            for i, (ipt, grd) in enumerate(tqdm(g.VL_LOADER, disable=not p.bar)):
                prd = g.MODEL(ipt)
                tot_loss, dist_loss, con_loss = g.CRIT(prd, grd)

                seq = fasta2str(ipt)
                grd_set = grd2set(grd)
                prd_set = prd2set(ipt=prd,
                                  seq=seq,
                                  thres_pairs=len(grd_set),
                                  min_dist=4,
                                  min_prob=.5)
                bin_scores = get_scores(prd_set, grd_set, len(seq))
                pcc, mae_l, mae_dist = get_dists(prd, grd, **p.dist_kwargs)
    
                raw_loss_tot += tot_loss.item()
                dist_loss_tot += dist_loss
                con_loss_tot += con_loss
                mcc_tot += bin_scores["mcc"]
                f1_tot += bin_scores["f1"]
                prd_pairs_tot += bin_scores["n_prd"]
                grd_pairs_tot += bin_scores["n_grd"]
                # dist_errors_pccs_tot += dist_errors_pccs
                ls.append(len(seq))
                pccs.append(pcc)
                mae_ls.append(mae_l)
                mae_dists.append(mae_dist)

                prd_grds.append((grd, prd))

        if epoch == p.save_epoch:
            if not p.out_file:
                out_time = int(time.time()) % 100_000
                p.out_file = f"{out_time}_vals.pt"
            torch.save(prd_grds, p.out_file)
            logging.info(f"\nSaving data as {p.out_file}\n")

        mcc_mean = mcc_tot / g.VL_LEN
        raw_loss_mean = raw_loss_tot / g.VL_LEN
        dist_loss_mean = dist_loss_tot / g.VL_LEN
        con_loss_mean = con_loss_tot / g.VL_LEN
        f1_mean = f1_tot / g.VL_LEN
        prd_pairs_mean = prd_pairs_tot / g.VL_LEN
        grd_pairs_mean = grd_pairs_tot / g.VL_LEN

        pcc_mean = torch.stack(pccs).mean(0).tolist()
        mae_l_mean = torch.stack(mae_ls).mean(0).tolist()
        mae_dist_mean = torch.stack(mae_dists).mean(0).tolist()
        # dist_errors_pccs_mean = (dist_errors_pccs_tot / g.VL_LEN).tolist()

        delta = datetime.datetime.now() - start
        mess = (f"Validation time: {delta.seconds}s\n" +
                f"Total validation loss: {raw_loss_mean}\n" +
                f"Contact validation loss: {con_loss_mean}\n" +
                f"Distance validation loss: {dist_loss_mean}\n" +
                f"MCC: {mcc_mean}\n" +
                f"L1 error by L: {mae_l_mean}\n" +
                f"PCC: {pcc_mean}\n" +
                f"L1 error by dist: {mae_dist_mean}\n" +
                f"Ground pairs: {grd_pairs_mean}\n" +
                f"Predicted pairs: {prd_pairs_mean}\n")
        if mcc_mean > g.BEST_MCC:
            g.BEST_MCC = mcc_mean
            g.BEST_LOSS = raw_loss_mean
            if p.vl_chk:
                torch.save({"epoch": epoch,
                            "model_state_dict": g.MODEL.state_dict(),
                            "optimizer_state_dict": g.OPT.state_dict(),
                            "mcc_history": g.BEST_MCC,
                            "grd_pairs_mean": grd_pairs_mean,
                            "model_state_dict": g.MODEL.state_dict(),
                            "optimizer_state_dict": g.OPT.state_dict(),
                            "mcc_history": g.BEST_MCC,
                            "grd_pairs_mean": grd_pairs_mean,
                            "prd_pairs_mean": prd_pairs_mean,
                            "mcc_mean": mcc_mean,
                            "f1_mean": f1_mean,
                            "raw_loss_mean": raw_loss_mean,
                            "dist_errors_mean": dist_errors_mean},
                           p.vl_chk)
            mess += f"*****NEW MAXIMUM MCC*****\n"
        mess += "\n\n"
        logging.info(mess)
        g.MODEL.train()
