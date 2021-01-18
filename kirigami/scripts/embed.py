import os
from tqdm import tqdm
import torch
from kirigami.nn.Embedding import *

def embed(in_path: str, out_path: str):
    os.path.exists(out_path) or os.mkdir(out_path)
    bp = BpseqEmbedding()
    with open(in_path, 'r') as f:
        in_files = f.read().splitlines()

    out_names = []
    for file in tqdm(in_files):
        with open(file, 'r') as f:
            txt = f.read()
        embed = bp(txt)
        base_name = os.path.basename(file)
        base_name, _ = os.path.splitext(base_name)
        out_name = os.path.join(out_path, base_name + '.pt')
        out_names.append(out_name)
        torch.save(embed, out_name)

    out_list = os.path.join(out_path, 'out.lst')
    with open(out_list, 'w') as f:
        for out_name in out_names:
            f.write(out_name + '\n')
