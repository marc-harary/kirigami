from argparse import Namespace
from pathlib import Path
from multipledispatch import dispatch
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import munch
from kirigami.utils.data_utils import *
import kirigami.nn


__all__ = ['predict']


@dispatch(Namespace)
def predict(args) -> None:
    config = path2munch(args.config)
    in_file = args.in_list
    return predict(config, in_list)


@dispatch(munch.Munch, Path):
def predict(config, in_list, out_list) -> None:
    '''Evaluates model from config file'''
    try:
        saved = torch.load(config.training.best)
    except os.path.exists(config.training.checkpoint):
        saved = torch.load(config.training.checkpoint)
    finally:
        raise FileNotFoundError('Can\'t find checkpoint files')

    model = MainNet(config.model)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()

    dataset = FastaDataset(in_file)
    with open(out_list, 'r') as f:
        lines = f.read().splitlines()

    for out_file, seq in tqdm(zip(lines, dataset)):
        pred = model(seq)
        pred = get_contacts(pred)
        bpseq_str = contacts2bpseq(pred)
        with open(out_file, 'w') as f:
            f.write(bpseq_str)
