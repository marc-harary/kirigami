import sys
import os
import argparse
from tqdm import tqdm
import torch

sys.path.append('..')
from kirigami.nn.Embedding import *


def main():
    parser = argparse.ArgumentParser('Embed bpseq files as `torch.Tensor` objects')
    parser.add_argument('--in_path', type=str, help='Path to input list file')
    parser.add_argument('--out_path', type=str, help='Path to output directory')
    args = parser.parse_args()

    os.path.exists(args.out_path) or os.mkdir(args.out_path)
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
        out_name = os.path.join(args.out_path, base_name + '.pt')
        out_names.append(out_name)
        torch.save(embed, out_name)

    out_list = os.path.join(args.out_path, 'out.lst')
    with open(out_list, 'w') as f:
        for out_name in out_names:
            f.write(out_name + '\n')


if __name__ == '__main__':
    main()
