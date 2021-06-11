import os
from pathlib import Path
from argparse import Namespace
from typing import List
import csv

from munch import Munch
from multipledispatch import dispatch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from kirigami.utils.data import BpseqDataset
from kirigami.utils.convert import path2munch, binarize, tensor2pairmap, get_scores, tensor2bpseq


__all__ = ["evaluate"]


@dispatch(Namespace)
def evaluate(args: Namespace) -> List[Path]:
    """Evaluates model from config file"""
    config = path2munch(args.config)
    return evaluate(config=config,
                    in_list=args.in_list,
                    out_dir=args.out_directory,
                    thres=args.thres,
                    quiet=args.quiet,
                    disable_gpu=args.disable_gpu)


@dispatch(Munch, Path, Path, float, bool, bool)
def evaluate(config: Munch,
             in_list: Path,
             out_dir: Path,
             thres: float,
             quiet: bool = False,
             disable_gpu: bool = False) -> List[Path]:
    """Evaluates model from config file"""

    try:
        saved = torch.load(config.training.best)
    except FileNotFoundError:
        saved = torch.load(config.training.checkpoint)

    device = DEVICE if not disable_cuda else torch.device("cpu")
    model = torch.nn.DataParallel(MainNet(config.model))
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()

    os.path.exists(out_dir) or os.mkdir(out_dir)
    bpseq_dir = os.path.join(out_dir, "bpseqs")
    os.path.exists(bpseq_dir) or os.mkdir(bpseq_dir)
    out_csv = os.path.join(out_dir, "scores.csv")

    with open(in_list, "r") as f:
        in_bpseqs = f.read().splitlines()
    out_bpseqs = []
    for in_bpseq in in_bpseqs:
        out_bpseq = os.path.basename(in_bpseq)
        out_bpseq = os.path.join(bpseq_dir, out_bpseq)
        out_bpseqs.append(out_bpseq)
    
    dataset_class = EmbeddedDataset if config.pre_embedded else BpseqDataset
    dataset = dataset_class(train_set,
                            shuffle=config.data.shuffle,
                            device=device,
                            batch_load=config.batch_load)
    loader = DataLoader(dataset, batch_size=config.data.batch_size)
    loop_zip = zip(out_bpseqs, loader)
    loop = loop_zip if quiet else tqdm(loop_zip)
    criterion = eval(config.training.criterion)

    fp = open(out_csv, "w")
    writer = csv.writer(fp)
    writer.writerow(["basename","loss","tp","fp","tn","fn","mcc","f1","ground_pairs","pred_pairs"])
    loss_tot, f1_tot, mcc_tot = 0., 0., 0.
    for out_bpseq, (sequence, ground) in loop:
        pred = model(sequence)
        loss = float(criterion(pred, ground))
        pred = binarize(pred, thres=thres)
        contact_map_pred, contact_map_ground = tensor2pairmap(pred), tensor2pairmap(ground)
        basename = os.path.basename(out_bpseq)
        basename, _ = os.path.splitext(basename)
        out = get_scores(contact_map_pred, contact_map_ground)
        f1_tot += out.f1
        mcc_tot += out.mcc
        loss_tot += loss
        writer.writerow([basename, loss, *list(out)])
        bpseq_str = tensor2bpseq(sequence, pred)
        with open(out_bpseq, "w") as f:
            f.write(bpseq_str+"\n")
    fp.close()

    if not quiet:
        length = len(loader)
        mean_loss = loss_tot / length
        mean_f1 = f1_tot / length
        mean_mcc = mcc_tot / length
        print(f"Mean loss for test set: {mean_loss}")
        print(f"Mean F1 score for test set: {mean_f1}")
        print(f"Mean MCC score for test set: {mean_mcc}")

    return out_bpseqs
