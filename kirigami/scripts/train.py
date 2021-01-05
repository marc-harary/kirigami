import sys
import argparse
import toml

import torch
from torch import nn

sys.append('../nn')
from OneHot import OneHot

parser = argparse.ArgumentParser()
parser.add_argument('-q', help='quiet', type=bool, default=False)
parser.add_argument('--conf', help='path to configuration file', type=str)
args = parser.parse_args()

with open(args.conf, 'r') as f:
    conf = toml.loads(f.read())

layers = conf['layers']
net = nn.Sequential(OneHot())

for layer in layers:
   layer_func = layer.pop("layer_type")
   try:
       layer_class = getattr(nn, layer_func)
   except:
       raise AttributeError("Invalid layer type") 
   layer_obj = layer_class(**layer) 
   net.append(layer_obj)
   net.append(act_func)
