import argparse
import json
import sys
sys.path.append('../nn')

import torch
from torch import nn

from MainNet import *


parser = argparse.ArgumentParser()
parser.add_argument('-q', help='quiet', type=bool, default=False)
parser.add_argument('--conf', help='path to configuration file', type=str)
args = parser.parse_args()

with open(args.conf, 'r') as f:
    conf = json.loads(f.read())

net = MainNet(conf['layers'])
