import argparse
import json
from munch import munchify
import torch
from torch import nn
from ..nn.MainNet import *

parser = argparse.ArgumentParser()
parser.add_argument('-q', help='quiet', type=bool, default=False)
parser.add_argument('--conf', help='path to configuration file', type=str)
args = parser.parse_args()

with open(args.conf, 'r') as f:
    conf_str = f.read()
    conf_dict = json.loads(conf_str)
    conf = munchify(conf_dict)

net = MainNet(conf.layers)